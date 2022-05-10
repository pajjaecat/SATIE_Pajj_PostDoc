""" Module that defines all the different variables that are used in simulations
"""

import pandas as pd, cvxpy as cp, numpy as np

#Optimization Temporal Values
Δt = 5                       #Temporal resolution of the study (in minutes)
ΔT  = 15//Δt                 #Imbalance Settlement Period (Definded related to the temporal resolution 
                             # i.e. 15 periods if Δt = 1mn or 3periods if Δt = 5mn )

# Frequencies as a string
ds_freq_str = str(Δt) + 'T'  # 5 mn


#Network Base and Nominal values
#Base Values 
v_base_sec = 0.416e3                                #Base voltage of secondary side of the distribution network
s_base = 0.315e6                                    #Base power of the distribution network
v_base_prim = 11e3                                  #Base voltage of primary side of the distribution network
z_base_sec = (v_base_sec*v_base_sec)/s_base         #Base impedance of the secondary side of the distribution network
z_base_prim = (v_base_prim*v_base_prim)/s_base      #Base voltage of primary side of the distribution network
#Nominal values
nom_voltage = 0.416e3      #Nominal Voltage
voltage_tolerance = 10         #Voltage tolerance (in %)
s_max = s_base      #Max apparent power  of the distribution network



#EVs Values
eff_chrg = 0.95        #Charging Efficiency of the EVs(in %)
eff_dischrg = 0.96     #Discharging Efficiency of the EVs(in %)
E_bat = 30e3           #Rated Energy Capacity of the EVs (in Wh)
p_max = 7              #Rated Active Power of the EVs (in kW)
q_max = 3              #Rated Reactive Power of the EVs (in kVAR) [NOT USED YET]
e_bat = E_bat          
soc_min = 0.3                 #Minimum SOC of the EVs
soc_max = 0.8                 #Maximum SOC of the EVs
soc_min_dep = 0.7             #Minimum SOC of the EVs at departure time
soc_max_arr = 0.55            #Maximum SOC of the EVs at arrival time
n_cycles=14000                                 #No. of total cycles of the EVs battery 
dod = 40/100                                   #Depth of discharge of the EVs battery 
e_tp = eff_dischrg*n_cycles*E_bat*dod          #"Throughput" of the EVs battery



#Forecasting Error Values
err_pv = 0.          #Error in the PV forecast
err_load = 0           #Error in the load forecast
err_ev = 0             #Error in the EV forecast

fullDay_min = 1440   # Total number of minute of a full day
per_day = fullDay_min//Δt   # Total number of periods in a day 
day2_min = 1* fullDay_min   # Starting minute of the second day

    
### Define MPC horizon dictionary 
# Dictionary to hold the Horizon in minute
hor_dict = {'15mn':15, 
            '30mn':30, 
            '01H':60, 
            '02H':60*2, 
            '04H':60*4, 
            '06H':60*6, 
            '08H':60*8, 
            '12H':60*12, 
            '24H':60*24}
    
    
# 'C:\\Users\\jprince\\Documents\\NUTSTORRE\\My_Jupiter\\GitJupLab\\SATIE_jupy\\MILP\\DataFiles\\EVs_Data_5_reduced3.csv'
root_folder = 'C:\\Users\\jprince\\Documents\\NUTSTORRE\\My_Jupiter\\GitJupLab\\SATIE_jupy\\'
evs_data_df = pd.read_csv(root_folder+'MILP\\DataFiles\\EVs_Data_5_reduced3.csv')


sld_freq = 1 # Sliding horizon frequency in minutes
