import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import streamlit as st
from tqdm import tqdm

# Scope:
# 4 Wheel drive model of RC Car. This will be used to spec a battery and motor for FRC. 
# Will try to incorporate as many nonlinearities as possible with the motor, battery, and controller
# This script will simulate a straightline acceleration at the limits of the battery, tires, motor, and aerodynamics
# Drag and Lift is considered as estimations. Bluff body drag will be considered

# TODO: Add preconfigured options to the classes to make testing easier. Along with custom option
# TODO: Learn how to make and import modules

# TODO: Better Charactize the Motor. Make Motor Efficiency map. Need to make other script for this
# TODO: Better Charactize the battery. Include temperature and other effects from manufacturer
# TODO: Further down the line but better understand the controller and integrate it into this script

class Vehicle:
    ''' 
    | a is distance from cg to front axle (meters)
    | b is distance from cg to rear axle (meters)
    | m is mass (kilograms)
    | h is cg height (meters)
    | l is wheelbase (meters)
    | cd is coefficent of drag
    | A is aeroreference area
    '''

    def __init__(self, a, b, m, h, cd, A):
        # -- Setting Vehicle Parameters --
        self.a = a
        self.b = b
        self.m = m
        self.h = h
        self.l = a + b
        self.cd = cd
        self.A = A

class Motor:
    ''' 
    | Kv is the rotational constant (RPM/Volt)
    | Kt is torque constant (N-m/Amp)
    | k is the motor efficency (System Efficency at the moment)
    '''
    def __init__(self,Kv,k):
        self.Kv = Kv
        self.Kt = 1/(Kv*0.10472)
        self.k = k # Efficency

class Battery:
    ''' 
    | Ah is the battery pack capacity (Ah)
    | C is the continous discharge current (C)
    | V is the nominal Voltage (V)
    | B_time is the time at burst current (s)
    '''
    def __init__(self,Ah,C,V,B_time):
        self.Ah = Ah
        self.C = C
        self.V = V

        self.Constant = C * Ah
        self.Burst = self.Constant * 1.95 # 2 would be the standard but the batteries are fused to 2 times the continous current
        self.Burst_time = B_time

class Tire:
    ''' 
    | r is the tire radius (meters) (Should be effective radius if known)
    | J is wheel inertia (kg-m^2)
    | a,b,c, and d are the pacejka coefficents
    | fudge is the decrease in friction force from the data at calspan to the road
    '''
    def __init__(self, r, J, a, b, c, d, fudge):
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.J = J
        self.fudge = fudge
    
    def lonForce(self,slip,normalForce):
        Fx = -normalForce * self.d * np.sin(self.c * np.arctan(self.b*slip - self.a * (self.b*slip - np.arctan(self.b*slip))))
        return Fx

class Sim:
    ''' 
    | This class is the simualtion object and can be called to run the longitudinal sim
    | When creating an instance of the sim. The vehicle, motor, tire, and battery are specificed
    '''
    # DONE: Convert to 4 Wheels
    # TODO: Use battery options to control the current limited condition, involve burst time by allowing burst up until the saturation of a counter
    def __init__(self,Vehicle,Motor,Tire,Battery,dt,runtime):
        self.Vehicle = Vehicle
        self.Motor = Motor
        self.Tire = Tire
        self.Battery = Battery
        self.dt = dt
        self.runtime = runtime

        # Defining the state variables / Intial Conditions
        self._current_x = 0
        self._current_x_dot = 0.0001
        self._current_x_ddot = 0
        self._current_time = 0
        self._current_amps = [0, 0, 0, 0]
        self._inital_Fz =  [-9.81 * Vehicle.m * Vehicle.b/Vehicle.l / 2, -9.81 * Vehicle.m * Vehicle.b/Vehicle.l / 2, 
                            -9.81 * Vehicle.m * Vehicle.a/Vehicle.l / 2, -9.81 * Vehicle.m * Vehicle.a/Vehicle.l / 2]
        self._current_Fz = [-9.81 * Vehicle.m * Vehicle.b/Vehicle.l / 2, -9.81 * Vehicle.m * Vehicle.b/Vehicle.l / 2, 
                            -9.81 * Vehicle.m * Vehicle.a/Vehicle.l / 2, -9.81 * Vehicle.m * Vehicle.a/Vehicle.l / 2]
        self._current_P = [0, 0, 0, 0]
        self._current_slip = [0, 0, 0, 0]
        self._current_w = [0, 0, 0, 0]

        # Output DataFrame
        self.column_names = ['time (s)', 
                            'x (m)', 
                            'x_dot (m/s)', 
                            'x_ddot (m/s/s)', 
                            'Right Front amps (A)', 'Left Front amps (A)', 'Left Rear amps (A)', 'Right Rear amps (A)', 
                            'FZ Right Front (N)', 'FZ Left Front (N)', 'FZ Left Rear (N)', 'FZ Right Rear (N)', 
                            'FX Right Front (N)', 'FX Left Front (N)', 'FX Left Rear (N)', 'FX Right Rear (N)', 
                            'Right Front Slip Ratio (-)', 'Left Front Slip Ratio (-)', 'Left Rear Slip Ratio (-)', 'Right Rear Slip Ratio (-)', 
                            'Right Front Wheel Speed (rad/s)', 'Left Front Wheel Speed (rad/s)', 'Left Rear Wheel Speed (rad/s)', 'Right Rear Wheel Speed (rad/s)']
        self.output = pd.DataFrame(columns = self.column_names)
    
    def update_values(self, x, x_dot, x_ddot, time, amps, Fz, P, slip, w):
        # This function updates the state variables
        self._current_x = x
        self._current_x_dot = x_dot
        self._current_x_ddot = x_ddot
        self._current_time = time
        self._current_amps = amps
        self._current_Fz = Fz
        self._current_P = P
        self._current_slip = slip
        self._current_w = w

    def w_fun(self, w, T, axle):
        # This function solves for the rotational velocity of the wheel given a driveline torque. This is used when the vehicle is amperage limited
        return self.Tire.J * (w - self._current_w[axle]) + self.Tire.r * self.Tire.lonForce((1-(self._current_x_dot/(self.Tire.r * w))),self._current_Fz[axle]) * self.Tire.fudge - T

    def accel_FrictionLimited(self):

        P_rf_list = []
        P_lf_list = []
        P_lr_list = []
        P_rr_list = []
        P = list(range(len(self._current_Fz)))
        T = list(range(len(self._current_Fz)))
        w = list(range(len(self._current_Fz)))
        amps = list(range(len(self._current_Fz)))
        slip = list(range(len(self._current_Fz)))
        Fz = list(range(len(self._current_Fz)))


        # Finding Peak Force Avaliable
        for n in range(100):
            slip_it = n/100
            P_rf_list.append(self.Tire.lonForce(slip_it,self._current_Fz[0]) * self.Tire.fudge)
            P_lf_list.append(self.Tire.lonForce(slip_it,self._current_Fz[1]) * self.Tire.fudge)
            P_lr_list.append(self.Tire.lonForce(slip_it,self._current_Fz[2]) * self.Tire.fudge)
            P_rr_list.append(self.Tire.lonForce(slip_it,self._current_Fz[3]) * self.Tire.fudge)

        # Finding Slip at Peak Force
        P[0] = max(P_rf_list)
        slip[0] = P_rf_list.index(P[0])/100

        P[1] = max(P_lf_list)
        slip[1] = P_lf_list.index(P[1])/100

        P[2] = max(P_lr_list)
        slip[2] = P_lr_list.index(P[2])/100

        P[3] = max(P_rr_list)
        slip[3] = P_rr_list.index(P[3])/100

        # Finding Wheel Speed
        for n in range(len(P)):
            w[n] = self._current_x_dot / (self.Tire.r * (1 - slip[n]))

        # Finding Driveline Torque Required
        for n in range(len(P)):
            T[n] = self.Tire.J * (w[n] - self._current_w[n]) + self.Tire.r * P[n]

        # Finding Amps required
        for n in range(len(P)):
            amps[n] = T[n] / self.Motor.Kt * (1 / self.Motor.k)

        # print('Torque Tire : ' + str(T[2]) + ' Slip Tire : ' + str(slip[2])) FIXME: Remove
        # print('WS Tire : ' + str(w[2])) FIXME: Remove

        # Setting Amperage Limit and back solving for other variables
        for n in range(len(self._current_Fz)):
            if amps[n] > self.Battery.Burst:
                amps[n] = self.Battery.Burst
                T[n] = amps[n] * self.Motor.Kt * (1 / self.Motor.k)
                guess = w[n] * 0.8
                w[n] = fsolve(self.w_fun,guess,args=(T[n], n))
                slip[n] = (self.Tire.r * w[n] - self._current_x_dot) / (self.Tire.r * w[n])
                P[n] = self.Tire.lonForce(slip[n],self._current_Fz[n]) * self.Tire.fudge

        # print('Torque Amp : ' + str(T[2]) + ' Slip Amp : ' + str(slip[2])) FIXME: Remove

        # Air Resistance
        F_aero = (self.Vehicle.cd * 1.225 * self._current_x_dot**2 * self.Vehicle.A) / 2

        # Longitudinal Acceleration
        x_ddot = (sum(P) - F_aero) / self.Vehicle.m 

        # New Velocity
        x_dot = self._current_x_dot + x_ddot * self.dt

        # New Position
        x = self._current_x + x_dot * self.dt

        # Weight Transfer
        wtfr = self.Vehicle.h / self.Vehicle.l * self.Vehicle.m * x_ddot / -9.81

        # New Normal Forces
        for n in range(len(P)):
            if n <= 1:
                Fz[n] = self._inital_Fz[n] - wtfr/2
            else: Fz[n] = self._inital_Fz[n] + wtfr/2

        time = self._current_time + self.dt

        return x, x_dot, x_ddot, time, amps, Fz, P, slip, w

    def __call__(self):
        for dt in tqdm(range(int(self.runtime/self.dt)),'Running Sim : '):
            current = self.accel_FrictionLimited()
            self.update_values(*current)
            output_data =  [self._current_time, 
                            self._current_x, 
                            self._current_x_dot, 
                            self._current_x_ddot, 
                            self._current_amps[0], self._current_amps[1], self._current_amps[2], self._current_amps[3],
                            self._current_Fz[0], self._current_Fz[1], self._current_Fz[2], self._current_Fz[3], 
                            self._current_P[0], self._current_P[1], self._current_P[2], self._current_P[3], 
                            self._current_slip[0], self._current_slip[1], self._current_slip[2], self._current_slip[3], 
                            self._current_w[0], self._current_w[1], self._current_w[2], self._current_w[3]]
            output = pd.DataFrame(np.array(output_data, dtype=object).reshape(-1,len(output_data)),columns = self.column_names)
            self.output = pd.concat([self.output,output], ignore_index=True)

# Defining Vehicle
frc_vehicle = Vehicle(0.126,0.126,5,0.032,0.75,0.0418)
frc_motor = Motor(2000,0.8)
frc_tire = Tire(0.032,0.00001667,1.0301,16.6675,0.05343,65.1759,0.5)
frc_battery = Battery(10,5,3.6,8)

# Creating Sim Object
sim = Sim(frc_vehicle,frc_motor,frc_tire,frc_battery,0.01,10)

# Running Sim
sim()

# Getting Output
data = sim.output

fig = plt.figure(figsize=plt.figaspect(4))
gs = fig.add_gridspec(4, hspace=0.33)
axs = gs.subplots()

axs[0].plot(data['Right Front Wheel Speed (rad/s)'], label = 'Right Front Wheel Speed (rad/s)')
axs[0].plot(data['Right Rear Wheel Speed (rad/s)'], label = 'Right Rear Wheel Speed (rad/s)')
axs[0].legend()

axs[1].plot(data['Right Front Slip Ratio (-)'], label = 'Right Front Slip Ratio (-)')
axs[1].plot(data['Right Rear Slip Ratio (-)'], label = 'Right Rear Slip Ratio (-)')
axs[1].legend()

# axs[2].plot(data['FZ Right Front (N)'], label = 'FZ Right Front (N)')
# axs[2].plot(data['FZ Right Rear (N)'], label = 'FZ Right Rear (N)')
axs[2].plot(data['FX Right Front (N)'], label = 'FX Right Front (N)')
axs[2].plot(data['FX Right Rear (N)'], label = 'FX Right Rear (N)')
axs[2].legend()
axs[2].set_ylim([0, 50])

axs[3].plot(data['x_dot (m/s)'], label = 'x_dot (m/s)')
axs[3].plot(data['x_ddot (m/s/s)'], label = 'x_ddot (m/s/s)')
axs[3].legend()


st.pyplot(fig)

# newslip = np.linspace(-1,1,200)

# Fz = np.repeat(-40,len(newslip))

# fx = frc_tire.lonForce(newslip,Fz) * frc_tire.fudge

# fig = plt.figure()
# plt.plot(newslip,fx)

# st.pyplot(fig)