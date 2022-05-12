""" 
    This Module contains all the functions used in the regime switching solar irradiance prediction 

"""


import pickle         # saving module
import probVar, brp  # My own modules 


# get numpy,  pandas and cvxpy  from probVar
np = probVar.np
pd = probVar.pd


# 1440: Length of a day | 5: Freq of daily irradiance data
day_totPeriod = 1440 //5  #Total period in a day



def zeroing(daily_data, per_start_daylight, per_end_daylight):
    """
    Set to zero all the data not in the interval [per_start_daylight, per_end_daylight]
    i.e. make sure all the data outside the dayligth period equals zero
    
    Inputs:
    -------
    daily_data: 1D (1*288) Float array
        Data of a whole day. 
    per_start_daylight: Int
        Starting period of daylight
    per_end_daylight: Int
        Ending period of daylight
        
    Output:
        Daily data
    
    """
    daily_data[daily_data<0]= 0
    daily_data[:per_start_daylight] = 0
    daily_data[per_end_daylight:] = 0
    
    return daily_data





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def initialize_predGHI (model_params, per_daylight, prevDay_data, pred_GHI, init_dayType=0, day_totPeriod=288):
    """
    Initialize the predicted GHI for the first day at the initial instant
    
    Inputs:
    -------
    model_params: Tuple
        Parameter to compute the predicted GHI
        (0) beta_dict : Dict 
                Parameters beta for each regime or daytype
        (1) alpha_dict : Dict 
                Parameters alpha for each regime or daytype
        (2) i_list: list
                Coefficients i in cos(iwt)
    per_daylight: Tuple
        Daylight periods
        (0) per_start_daylight: Int
                Starting period of daylight
        (1) per_end_daylight: Int
                Ending period of daylight
    prevDay_data: Dataframe
        Previous Day data i.e. 'Clearsky GHI' and 'GHI'. 
        Only the 'Clearsky GHI', is used here.
    pred_GHI: float array ((nb_sim_days+1)*288), Modified in place
        Predicted GHI over the number of day of simulation
    init_dayType: Int, Default 0
        The initial regime or day type
    day_totPeriod: Int, default 288
        Number of period in a whole day since the input data is at a freq of 5mn
 
    
    Output:
    ------
        pred_GHI updated for the first day
    
    """
    
    #Unpack inputs 
    beta_dict, alpha_dict, i_list = model_params 
    per_start_daylight, per_end_daylight = per_daylight
    
    T = per_end_daylight - per_start_daylight # Total number of period of dailight 
    w = 2*np.pi/T # T must define a whole cosine period
    
    t = np.arange(0,day_totPeriod)

    # Compute predcited GHI of the previous day based on its 'Clearsky GHI'
    prevDay_regimeData = (beta_dict[init_dayType][0] + beta_dict[init_dayType][1]*prevDay_data['Clearsky GHI'].values +
                      np.sum([alphas*np.cos(i*w*t) for i, alphas in zip(i_list,alpha_dict[init_dayType])],axis=0))

    #Set the predicted values of the current day supposing the daytype is given by init_dayType
    pred_GHI[:day_totPeriod] = zeroing(prevDay_regimeData, per_start_daylight, per_end_daylight) 
    
    return pred_GHI





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def dayType_dataPred(model_params, per_daylight, curDay_data ):
    """
    Compute the Irradiance prediction value according to the type of day (regime)
    
    Inputs:
    -------
    model_params: Tuple
        Parameter to compute the predicted GHI
        (0) beta_dict : Dict 
                Parameters beta for each regime or daytype
        (1) alpha_dict : Dict 
                Parameters alpha for each regime or daytype
        (2) i_list: list
                Coefficients i in cos(iwt)
    per_daylight: Tuple
        Daylight periods
        (0) per_start_daylight: Int
                Starting period of daylight
        (1) per_end_daylight: Int
                Ending period of daylight

    curDay_data: Dataframe
        Dataframe contraining the data (GHI and	Clearsky GHI) of the 
        current day
        
    Outputs:
    -------
    pred_data_dict: Dict
        Dictionnary containing the predictec GHI dependent on the type of day.
        Dict index: 
            0 ==> Clear day
            1 ==> Overcast day
            2 ==> Mild day 
            3 ==> Moderate day
            4 ==> High day
    """
    
    #Unpack inputs 
    beta_dict, alpha_dict, i_list = model_params 
    per_start_daylight, per_end_daylight = per_daylight
    
    day_totPeriod = 1440 //5 
    T = per_end_daylight - per_start_daylight # Total number of period of dailight 
    w = 2*np.pi/T # T must define a whole cosine period
    t = np.arange(0,day_totPeriod)


    
    pred_data_dict = {}
    for day_type in range(0,4):
        # Get parameters for the current day type
        ββ = beta_dict[day_type]
        αα = alpha_dict[day_type]

        # compute the predicted values dependent on the day type
        pred_data =  (ββ[0] + ββ[1]*curDay_data['Clearsky GHI'].values +
              np.sum([alphas*np.cos(i*w*t) for i, alphas in zip(i_list,αα)],axis=0))

        pred_data_dict.update({day_type: zeroing(pred_data,per_start_daylight, per_end_daylight)})
        
        
    return pred_data_dict





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def pred_Irradiance_updater(cur_k, lb_curDay, curDay_frcst_per, per_back_windows, prob_predValues, per_daylight, input_data, cur_regime):
    """
    Update The predicted irradiance, 
    
    Inputs:
    -------
    cur_k: Int
        Current simulation step k 
    lb_curDay: Int 
        Lower bound current day, i.e. d0_index
    curDay_frcst_per: Int list
        List of instant where the irradiance must 
        be updated over the current day
    per_back_windows: Int
        Number of period covered by the backward 
        looking windows to compute the MSE
    prob_predValues: 1D float array ((nb_days+1)*288)
        Predicted vector of the total number of days
    per_daylight: Tuple
        Variables associated with daylight period
        (0) per_start_daylight: Int 
                Starting daylight period
        (1) per_endt_daylight: Int 
                Ending daylight period
    inputs_data: Tuple
        Variables related to the current day
        (0) curDay_data: Dataframe
                Dataframe describing the currend day's Clearsky 
                GHI and GHI.
        (1) curDay_data_dict: Dict
                Predcited GHI for each type of day type. output of 
                the function << regSwitchSolarPred.dayType_dataPred(args) >>     
    cur_regime: Int
        Current regime 
        
        
    Output
    ------
    
    cur_regime :Int
        Current regime updated
    
    """
    # Unpack inputs
    per_start_daylight, per_end_daylight = per_daylight
    curDay_data, curDay_data_dict = input_data
    
    day_totPeriod = 1440 //5 # 1440 : Total mimutes of a day
    
    upd_cur_k = cur_k-lb_curDay # Updated current k so that the updated cur_k is always in [0, 288], Δt = 5 mn

    if cur_k in curDay_frcst_per :  # If updating instant ==> Recompute MSE and update given the regime producing the lowest MSE
        err_list = [] # list to store MSE
        
        for day_type in range(0,4): # Compute mean square error between actual irradiance and the current predicted 

#             datta = zeroing(curDay_data, per_start_daylight, per_end_daylight).loc['GHI'] - curDay_data_dict[day_type] # Extract Day 
            datta = zeroing(curDay_data['GHI'].values, per_start_daylight, per_end_daylight) - curDay_data_dict[day_type] # Extract Day 
            err = (datta[upd_cur_k-per_back_windows: upd_cur_k]**2).sum()/per_back_windows # Compute Mean squared error 
                                                                                           # usinng the back looking windows lenght
            err_list.append( err ) # Append error to a list 

        cur_regime = np.array(err_list).argmin() # Get index of the element that has produced lowest MSE
        #update predictec values for the future using the regime that has produced the lowest MSE
        prob_predValues[cur_k: cur_k+ day_totPeriod] = prop_curDayToHor(upd_cur_k, day_totPeriod, curDay_data_dict[cur_regime])
        
    else: # Otherwise Keep using the same regime as previous step
        prob_predValues[cur_k: cur_k+ day_totPeriod ] = prop_curDayToHor(upd_cur_k, day_totPeriod, curDay_data_dict[cur_regime])
                
                
    return prob_predValues, cur_regime





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def prop_curDayToHor(upd_cur_k, daily_totPeriod, curDay_data_dict_indx ):# Propagate current day to fill the horizon of one day
    """ 
    Update the predicted GHI (Global Horizontal irradiance)  produced by the regime with the lowest MSQ over 
    a period of 24h (maximum prediction period of MPC)  
    
    
    Inputs:
    -------
    upd_cur_k: Int
        Updated current k so that the updated cur_k is always in [0, day_totPeriod], Δt = 5 mn
    daily_totPeriod: Int
        Daily total horizon
    curDay_data_dict_indx: float Array (1*daily_totPeriod)
        Irradiance prediction of the regime that has produced the lowest MSE
        
        
    Output:
    -------
        pred_val_daily_totPeriod: float Array (1*daily_totPeriod)
        
    """
    
    pred_val_daily_totPeriod = np.zeros((day_totPeriod)) # create a vatiable to store the predicted value of the current
                                               # regime of P_PV irradiance over a period of one day
        
    remain_hor = day_totPeriod - upd_cur_k # Remaining Horizon to cover  i.e if upd_cur_k = 200,
    # ==> remain_hor = 288 - 200 = 88  
#     fill the predicted value over the horizon of one day
    pred_val_daily_totPeriod[:remain_hor] = curDay_data_dict_indx[upd_cur_k:]  
    pred_val_daily_totPeriod[remain_hor:] = curDay_data_dict_indx[:upd_cur_k]  
    
    
    return pred_val_daily_totPeriod
    
    
    
    
    
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________      
    
def definePpv_PlanReal(nb_sim_days, star_date, engagement_regime, file_loc, scale=150):
    """
    Define the real solar production and the planned (engagement) solar prediction. 
    The engagement is based on a certain fixed regime of day defined by engagement_regime.
    
    Inputs: 
    ------
    nb_sim_days: Int 
        Number of simulation days
    Star_date: String in format '2020 month day'
        First day to consider, i.e '2020 03 5'
    engagement_regime: Int 
        Regime of GHI prediction to use as P_pv_Plan
        # TODO: Engage on the main regime visited by the 
          model on previous day ( its daylight period to be more precise) 
            0 ==> Clear day
            1 ==> Overcast day
            2 ==> Mild day 
            3 ==> Moderate day
    file_loc: File 
        Define the location of the file containing the regime switching 
        irradiance parameters
    scale: Float, default: 150
        Value use to upscale (to convert) irradiance in power 
        
    Output:
    ------
    Ppv_PlanReal: Dataframe
        Dataframe containing the actual and the planned solar irradiance
    
    """

    daily_totPeriod = 1440 //5 # 1440 Total minutes in a day
    
    
    # BRP engagement and real Irradiance
    engagement_GHI = np.zeros((nb_sim_days*daily_totPeriod))
    real_GHI = np.zeros((nb_sim_days*daily_totPeriod))  
    
    # Define all the days to consider
    DAYS = pd.date_range(start=star_date, periods=nb_sim_days, freq='D')
    
    # Load file containig parameters 
    file_to_read = open(file_loc, "rb")
    extracted_data = pickle.load(file_to_read)
    
    # Extract data from file
    data2020 = extracted_data['2020_data']

    for day_index, cur_day in enumerate(DAYS):

        d0_index = (day_index)*daily_totPeriod       # lower bound current day
        d1_index = (day_index+1)*daily_totPeriod     # Upper bound currend day

        # Define days 
        day_init = cur_day
        day_future = cur_day + cur_day.freq

        #Extract data of current day
        curDay_data = data2020[(data2020.index >= day_init) & (data2020.index < day_future)]

        # Define predicted values dependant on the type of regime for all daytype
        curDay_data_dict = dayType_dataPred(extracted_data['model_params'], extracted_data['per_daylight'], curDay_data )


        real_GHI[d0_index:d1_index] = curDay_data['GHI'].values
        engagement_GHI[d0_index:d1_index] = curDay_data_dict[engagement_regime]
        
    
    # Dataframe index creation
    star_date = DAYS[0]
    date_index = pd.date_range(start=star_date, periods=nb_sim_days*daily_totPeriod, freq='5T')

    in_data = np.array([engagement_GHI, real_GHI]).T #Define Dataframe inputs
    Ppv_PlanReal = pd.DataFrame(data= in_data/scale, index=date_index, columns=['engagement_GHI','real_GHI'])
    
    
    return Ppv_PlanReal







