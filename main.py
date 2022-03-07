import numpy as np
import pandas as pd

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
        self._current_Fz = [Vehicle.m * Vehicle.b/Vehicle.l, Vehicle.m * Vehicle.a/Vehicle.l]
        self._current_P = [0, 0]
        self._current_slip = [0, 0]
        self._current_w = [0, 0]

        # Output DataFrame
        column_names = ['time (s)', 'x (m)', 'x_dot (m/s)', 'x_ddot (m/s/s)', 'Front amps (A)', 'Rear amps (A)', 'FZ Front (N)', 'FZ Rear (N)', 'FX Front (N)', 'FX Rear (N)', 'Front Slip Ratio (-)', 'Rear Slip Ratio (-)', 'Front Wheel Speed (rad/s)', 'Rear Wheel Speed (rad/s)']
        output = pd.DataFrame(columns = column_names)
    
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

    def accel_FrictionLimited(self):

        P_f_list = []
        P_r_list = []

        # Finding Peak Force Avaliable
        for slip_it in range(100):
            slip_it = slip_it/100
            P_f_list.append(self.Tire.lonFriction(slip_it,self._current_Fz[0]) * self._current_Fz[0])
            P_r_list.append(self.Tire.lonFriction(slip_it,self._current_Fz[1]) * self._current_Fz[1])

        P_f = max(P_f_list)
        P_f_slip = P_f_list.index(P_f)

        P_r = max(P_r_list)
        P_r_slip = P_r_list.index(P_r)

        slip = [P_f_slip, P_r_slip]

        # Finding Wheel Slip Difference Needed
        w_f = self._current_x_dot / (self.Tire.r * (1 - P_f_slip))
        w_r = self._current_x_dot / (self.Tire.r * (1 - P_r_slip))

        w = [w_f, w_r]

        # Finding Driveline Torque Required
        T_f = self.Tire.J * (w_f - self._current_w[0]) + self.Tire.r * P_f
        T_r = self.Tire.J * (w_r - self._current_w[1]) + self.Tire.r * P_r

        # Finding Amps required
        amps_f = T_f / self.Motor.Kt * (1 / self.Motor.k)
        amps_r = T_r / self.Motor.Kt * (1 / self.Motor.k)

        amps = [amps_f, amps_r]

        # Longitudinal Acceleration
        x_ddot = (P_f + P_r) / self.Vehicle.m 

        # New Velocity
        x_dot = self._current_x_dot + x_ddot * self.dt

        # New Position
        x = self._current_x + x_dot * self.dt

        # New Normal Forces
        Fz_f = (self.Vehicle.m * 9.81 * self.Vehicle.b - (P_f + P_r) * self.Vehicle.h) / self.Vehicle.l
        Fz_r = (self.Vehicle.m * 9.81 * self.Vehicle.a + (P_f + P_r) * self.Vehicle.h) / self.Vehicle.l

        Fz = [Fz_f, Fz_r]

        time = self._current_time + self.dt

        P = [P_f, P_r]

        return x, x_dot, x_ddot, time, amps, Fz, P, slip, w

    def __call__(self):
        for dt in range(int(self.runtime/self.dt)):
            current = self.accel_FrictionLimited()
            self.update_values(*current)



frc = Vehicle(0.126,0.126,5,0.032)
frc_tire = Tire(0.032,0.00001667,1.0301,16.6675,0.05343,65.1759)
frc_motor = Motor(500,0.8)


sim = Sim(frc,frc_motor,frc_tire,0.01,5)

sim()