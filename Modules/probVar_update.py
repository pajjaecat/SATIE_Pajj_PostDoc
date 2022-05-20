# Module used to update all the global variable in probVar

import probVar
import pandas as pd

def Δt(Δt = 5): #Update all the variable related to Δt
    """
    Update all the variables related to Δt (the sampling frequency) 
    whithin the module <<probVar>>

    Inputs:
    -------
    Δt : int
        Samplin frequency

    """
    # Update Δt and all its dependant variables
    
    probVar.Δt  = Δt
    probVar.ΔT  = 15//Δt
    probVar.ds_freq_str = str(Δt) + 'T'
    probVar.per_day = probVar.fullDay_min//Δt

    
    
def read_EV_df (df_location = "../DataFiles/EVsData/EVs_Data_5_reduced3.csv" ):
    """
    Read Ev_data_df from the location given as input

    Inputs:
    -------
    df_location : String, default in "../DataFiles/EVsData/EVs_Data_5_reduced3.csv"
        Ev_data_dataframe location

    """
    
    probVar.evs_data_df = pd.read_csv(df_location)
    if df_location == '../DataFiles/EVsData/EVs_Data_5_reduced3.csv':
        print('Ev_data_dataframe updated with the DEFAULT file located at', df_location)
    else :
        print('Ev_data_dataframe updated with the NEW file located at', df_location)