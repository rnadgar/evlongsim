import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import streamlit as st
from tqdm import tqdm

# The objective of this script is to be able to create a motor efficiency map (speed vs torque vs efficieny) based off basic motor information
# All inputs will be based off this motor initially Kv-1350 : https://store.tmotor.com/goods.php?id=1177
# Map is being created based off this article : https://things-in-motion.blogspot.com/2019/03/basic-bldc-pmsm-efficiency-and-power.html

class Motor:
    ''' 
    Inputs
    | Kv is the rotational constant (RPM/Volt)
    | wR is the winding resistance (Ohms)
    | maxA is the max current aa
    | Pc is the pole count
    | m is the mass of the motor (kg)
    '''
    def __init__(self,Kv,maxA,wR,Pc,m):
        self.Kv = Kv
        self.Kt = 1/(Kv*0.10472) # Torque constant (N-m/Amp)
        self.wR = wR
        self.Pc = Pc
        self.Sm = m * 0.3 # See article (this is an estimate) Sm = Stator Mass (kg)

rc_motor = Motor(1350,60,0.061,14,0.058)