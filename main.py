import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Vehicle:
    ''' 
    | a is distance from cg to front axle (meters)
    | b is distance from cg to rear axle (meters)
    | m is mass (kilograms)
    | h is cg height (meters)
    | l is wheelbase (meters)
    '''

    def __init__(self, a, b, m, h):
        # -- Setting Vehicle Parameters --
        self.a = a
        self.b = b
        self.m = m
        self.h = h
        self.l = a + b

class Motor:
    ''' 
    | Kv is the rotational constant (RPM/Volt)
    | Kt is torque constant (N-m/Amp)
    '''
    def __init__(self,Kv,k):
        self.Kv = Kv
        self.Kt = 1/(Kv*0.10472)
        self.k = k # Effiency
        
class Tire:
    def __init__(self, r, J, a, b, c, d):
        self.r = r
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.J = J
    
    def lonFriction(self,slip,normalForce):
        Fx = normalForce * self.d * np.sin(self.c * np.arctan(self.b*slip - self.a * (self.b*slip - np.arctan(self.b*slip))))
        u = Fx/normalForce
        return u

class Sim:
    
    def __init__(self,Vehicle,Motor,Tire,dt,runtime):
        self.Vehicle = Vehicle
        self.Motor = Motor
        self.Tire = Tire
        self.dt = dt
        self.runtime = runtime

        # Defining internal variables / Intial Conditions
        self._current_x = 0
        self._current_x_dot = 0.0001
        self._current_x_ddot = 0
        self._current_time = 0
        self._current_amps = [0, 0]
        self._inital_Fz = [Vehicle.m * Vehicle.b/Vehicle.l, Vehicle.m * Vehicle.a/Vehicle.l]
        self._current_Fz = [Vehicle.m * Vehicle.b/Vehicle.l, Vehicle.m * Vehicle.a/Vehicle.l]
        self._current_P = [0, 0]
        self._current_slip = [0, 0]
        self._current_w = [0, 0]

        # Output DataFrame
        column_names = ['time (s)', 'x (m)', 'x_dot (m/s)', 'x_ddot (m/s/s)', 'Front amps (A)', 'Rear amps (A)', 'FZ Front (N)', 'FZ Rear (N)', 'FX Front (N)', 'FX Rear (N)', 'Front Slip Ratio (-)', 'Rear Slip Ratio (-)', 'Front Wheel Speed (rad/s)', 'Rear Wheel Speed (rad/s)']
        self.output = pd.DataFrame(columns = column_names)
    
    def update_values(self, x, x_dot, x_ddot, time, amps, Fz, P, slip, w):
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
        return self.Tire.J * (w - self._current_w[axle]) + self.Tire.r * self.Tire.lonFriction((1-(self._current_x_dot/(self.Tire.r * w))),self._current_Fz[axle]) * self._current_Fz[axle] - T

    def accel_FrictionLimited(self):

        P_f_list = []
        P_r_list = []

        # Finding Peak Force Avaliable
        for slip_it in range(100):
            slip = slip_it/100
            P_f_list.append(self.Tire.lonFriction(slip,self._current_Fz[0]) * self._current_Fz[0])
            P_r_list.append(self.Tire.lonFriction(slip,self._current_Fz[1]) * self._current_Fz[1])

        P_f = max(P_f_list)
        P_f_slip = P_f_list.index(P_f)/100

        P_r = max(P_r_list)
        P_r_slip = P_r_list.index(P_r)/100

        # Finding Wheel Speed
        w_f = self._current_x_dot / (self.Tire.r * (1 - P_f_slip))
        w_r = self._current_x_dot / (self.Tire.r * (1 - P_r_slip))

        # Finding Driveline Torque Required
        T_f = self.Tire.J * (w_f - self._current_w[0]) + self.Tire.r * P_f
        T_r = self.Tire.J * (w_r - self._current_w[1]) + self.Tire.r * P_r

        # Finding Amps required
        amps_f = T_f / self.Motor.Kt * (1 / self.Motor.k)
        amps_r = T_r / self.Motor.Kt * (1 / self.Motor.k)

        # print('friction' + str(w_r))

        # Setting Amperage Limit and back solving for other variables
        if amps_f > 50:
            amps_f = 50
            T_f = amps_f * self.Motor.Kt * (1 / self.Motor.k)
            w_f = fsolve(self.w_fun,1,args=(T_f, 0))
            P_f_slip = 1 - self._current_x_dot / (self.Tire.r * w_f)
            P_f = self.Tire.lonFriction(P_f_slip,self._current_Fz[0]) * self._current_Fz[0]

        if amps_r > 50:
            amps_r = 50
            T_r = amps_r * self.Motor.Kt * (1 / self.Motor.k)
            w_r = fsolve(self.w_fun,1,args=(T_r, 1))
            # print('amp' + str(w_r))
            P_r_slip = 1 - self._current_x_dot / (self.Tire.r * w_r)
            P_r = self.Tire.lonFriction(P_r_slip,self._current_Fz[1]) * self._current_Fz[1]

        amps = [amps_f, amps_r]
        w = [w_f, w_r]
        slip = [P_f_slip, P_r_slip]
        P = [P_f, P_r]

        # Air Resistance
        F_aero = (0.75 * 1.225 * self._current_x_dot**2 * 0.0418) / 2
        print(F_aero)

        # Longitudinal Acceleration
        x_ddot = (P_f + P_r - F_aero) / self.Vehicle.m 

        # New Velocity
        x_dot = self._current_x_dot + x_ddot * self.dt

        # New Position
        x = self._current_x + x_dot * self.dt

        # Weight Transfer
        wtfr = self.Vehicle.h / self.Vehicle.l * self.Vehicle.m * x_ddot / 9.81

        # New Normal Forces
        Fz_f = self._inital_Fz[0] - wtfr
        Fz_r = self._inital_Fz[1] - wtfr

        Fz = [Fz_f, Fz_r]

        time = self._current_time + self.dt

        return x, x_dot, x_ddot, time, amps, Fz, P, slip, w

    def __call__(self):
        for dt in range(int(self.runtime/self.dt)):
            current = self.accel_FrictionLimited()
            self.update_values(*current)
            column_names = ['time (s)', 'x (m)', 'x_dot (m/s)', 'x_ddot (m/s/s)', 'Front amps (A)', 'Rear amps (A)', 'FZ Front (N)', 'FZ Rear (N)', 'FX Front (N)', 'FX Rear (N)', 'Front Slip Ratio (-)', 'Rear Slip Ratio (-)', 'Front Wheel Speed (rad/s)', 'Rear Wheel Speed (rad/s)']
            output_data = [self._current_time, self._current_x, self._current_x_dot, self._current_x_ddot, self._current_amps[0], self._current_amps[1], self._current_Fz[0], self._current_Fz[1], self._current_P[0], self._current_P[1], self._current_slip[0], self._current_slip[1], self._current_w[0], self._current_w[1]]
            output = pd.DataFrame(np.array(output_data, dtype=object).reshape(-1,len(output_data)),columns = column_names)
            self.output = pd.concat([self.output,output], ignore_index=True)

frc = Vehicle(0.126,0.126,5,0.032)
frc_tire = Tire(0.032,0.00001667,1.0301,16.6675,0.05343,65.1759)
frc_motor = Motor(2000,0.8)

sim = Sim(frc,frc_motor,frc_tire,0.01,20)

sim()

data = sim.output

plt.figure()
# plt.plot(data['Front Wheel Speed (rad/s)'], label = 'Front Wheel Speed (rad/s)')
# plt.plot(data['Rear amps (A)'], label = 'Rear amps (A)')
plt.plot(data['x_dot (m/s)'], label = 'x_dot (m/s)')
plt.legend()
plt.show()