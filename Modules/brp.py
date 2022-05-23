# My own module
import probVar


# get numpy and pandas from probVar
np = probVar.np
pd = probVar.pd




def create_probvar (t_sim, totBus_network, ev_data_dict ):
    """ Define all the variables to use over the problem horizon and 
    initialize the ones related to storage (soc_prob, soh_prob)
    NB: Add new variables to this function 
    
    Inputs:
    ------
    t_sim: Int
        Total duration of the simulation 
    totBus_network : Int
        Total number of Bus in the network
    ev_data_dict: Dict
        Dictionnoary containnig all the parameters of the Evs connected 
        to the network;  Output of function << brp.init_evDataParams(args) >>
        
    Output:
    ------
        All the created variables as tuple over the problem horizon such that
            (0) Active power related Variables
                ==> p_var = (p_ev_chrg_prob, del_p_ev_prob, p_bus_prob, p_gen_prob)
            (1) Ev Storage related variables
                ==> sto_var = (soc_prob, soc_pred_ins, soh_prob)
            (2) Network related variables
                ==> line_var = (theta_prob, voltage_prob )
             
    """

    
    #Variables for Active power decisions
    p_ev_chrg_prob = {} #EV Active Charging  power,
    del_p_ev_prob = {}  #Difference in planned EV active variable

    #Storage variables
    soc_prob = {} #State of charge 
    soc_pred_ins = {} #Predicted State of charge of all Ev at each instant over the mpc horizon
    # indexed by (cur_bus, k)
    soh_prob = {} # #State of health
    
    # Line variables
    theta_prob = np.zeros((totBus_network+1,t_sim))
    voltage_prob = np.zeros((totBus_network+1,t_sim))
    p_bus_prob = np.zeros((totBus_network+1,t_sim))
    # q_bus_prob = np.zeros((totBus_network+1,t_sim)) 
    p_gen_prob = np.zeros((1,t_sim))
    # q_gen_prob = np.zeros((1,t_sim))


    for cur_bus in probVar.evs_data_df['Bus'].values: # for each bus with ev
        # Active power 
        p_ev_chrg_prob.update({cur_bus: (np.zeros((1,t_sim))) })
        del_p_ev_prob.update({cur_bus: (np.zeros((1,t_sim))) })
       
        # Storage variables
        soc_prob.update({cur_bus: (np.zeros((1,t_sim+1))) })
        soh_prob.update({cur_bus: (np.zeros((1,t_sim+1))) })

        # Initialize soh and soc (initial instant i.e. arrival at first interval
        soh_prob[cur_bus][0,0] = ev_data_dict[cur_bus]['soh_arr1']
        soc_prob[cur_bus][0,0] = ev_data_dict[cur_bus]['soc_arr1']


    p_var = (p_ev_chrg_prob, del_p_ev_prob, p_bus_prob, p_gen_prob)
    sto_var = (soc_prob, soc_pred_ins, soh_prob)
    line_var = (theta_prob, voltage_prob )


    return p_var, sto_var,line_var 





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def def_ppv_plan (input_data_df, p_pv_f, plan_type=0, err_pv_f=0.15, avr_freq=60):# Define P_pv_plan
    """Define the planed P_pv depending on the input plan_type
    
    Inputs:
    -------
    input_data_df : Dataframe 
        Load data dataframe
    p_pv_f : dict
        Dictionary of P_pv for all users
    plan_type : Int
        0 ==>  Use the real biased Ppv ==> (P_pv_plan = Ppv + Ppv*err_pv_f) 
        1 ==>  Use the persistance model defined as previous day data
    err_f : float
        value of prediction error when using the plan type 0
    avr_freq : Int
        Number of minutes to consider to average the solar prediction, 60=>1Hour

        
    Outputs: 
    --------
    p_pv_plan : Dictionary containing the planed p_pv for each active bus 
    in the  load data dataframe.
    """
    
    p_pv_plan_f = {}# Define a dict to store the plan_ppv
    
    for i in input_data_df.index:
        bus_nbr = input_data_df.loc[i]['Bus'] #Get bus number at row i in the dataframe
        
        # Ppv_plan is based on the real pv + some error terms
        if plan_type == 0: 
            var1 = (p_pv_f[bus_nbr] + p_pv_f[bus_nbr]*err_pv_f)
            
        elif plan_type == 1:  
            # Ppv_plan is based previous day value: Persistance model
            # The first day is used as persistance model for the second day, so hence
            # the simulation starts at the second day

            # create persistance model by rolling data forwards and zeroing data of first day 
            roll_len = int(probVar.day2_min/probVar.Δt)              # Get how many steps to roll forward   
            var1 = np.roll(p_pv_f[bus_nbr],roll_len) # roll
            var1[:roll_len] = 0                      # Fill first day data in the rolled array
                                                     # with 0 
            var1 = var1 + var1*err_pv_f
        else: 
            raise ValueError('Wrong plan_type, Must be 0 or 1')
            
        
        pv_values = np.zeros((len(var1))); # Intermediate variable

        n = int(avr_freq/probVar.Δt) #60 minutes - 1 hour avg of solar prediction
        for j in range(0,len(var1),n):
            pv_values[j:j+n] = np.mean(var1[j:j+n])

        # Planned Load 
        p_pv_plan_f.update({bus_nbr: pv_values})
        
    
    return p_pv_plan_f





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def periodToHod (input_per = 0 ):# period to hour of day
    """
    Convert the input period in Hour of Day
    
    Inputs:
    -------
    input_per : Int
        Period 

        
    outputs: 
    -------
    out_hod : String
        Hour of day 
    """
    x = input_per
    out_str = (str((x*probVar.Δt)//probVar.fullDay_min) + 'D: '
               + str(((x*probVar.Δt)//60)%24) + 'h'+ str((x*probVar.Δt)%60))
    
    return out_str





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def init_arrDep_instants(nb_sim_days, nb_arr_dep, tot_bldg, arrDep_dict,  chrg_at_work=True ):
    """
   Initialize all the arrival and departure instants for all EVs
    
    Inputs:
    -------
    nb_sim_days: Int
        Number of days of simulation
    nb_arr_dep: Int
        Total number of arrival and departure during all the simulation
    tot_bldg: Int
        Total number of bulding considered in the network 
    arrDep_dict: Dict
        Dictionnary holding the min, mean, and max variable associated with 
        arrival and departure period. 
    chrg_at_work : Binary  (True by default)
        Wether charging at work must be considered or not

        
    outputs: 
    -------
    arr_dict_f : Dictionnary 
        Arrival instant dictionnary for all EVs
        
    dep_dict_f : Dictionnary 
        Dep instant dictionnary for all EVs
    """

    
    
    # create a lambda function to generate random departure and arrival time
    rand_hour = lambda hour,nb_pts : np.round(np.random.normal(loc=hour,scale=100/probVar.Δt,size=nb_pts), decimals=0)


    arr_dict_f = {} # Create a dict to save arrival data of cars during simulation
    dep_dict_f = {} # Create a dict to save departure data 

    # Initialize dict for each arrival and departure instants, this for all building considered 
    # in the electrical network even though all of them do not have an EV associated
    for i in range(1, nb_arr_dep+1): 
        arr_dict_f.update({i: np.zeros((tot_bldg))})
        dep_dict_f.update({i: np.zeros((tot_bldg))})
    
    
    cur_arr_dep = 1  # Current arrival and departure interval 
    
    # First arrival is considered to be the starting instant of the simulation k=0
    # First departure
    dep_dict_f[cur_arr_dep] = np.maximum(arrDep_dict['min_dep'] ,rand_hour(arrDep_dict['mean_dep'] ,tot_bldg)) 
    
    # For each simulation day
    for cur_day in range(0, nb_sim_days):
        var = cur_day*probVar.fullDay_min/probVar.Δt 
        # For curent day, Make sure arrival time > to previous departure time 
        while (dep_dict_f[cur_arr_dep] < arr_dict_f[cur_arr_dep+1]).sum() < tot_bldg :     
            dep_dict_f[cur_arr_dep] = np.minimum(arrDep_dict['max_dep'] + var,
                                                 np.maximum(arrDep_dict['min_dep'] + var, 
                                                            rand_hour(arrDep_dict['mean_dep'] + var ,tot_bldg) ) ) 
            arr_dict_f[cur_arr_dep+1] = np.minimum(arrDep_dict['max_arr_work'] + var, 
                                                   np.maximum(arrDep_dict['min_arr_work'] + var,
                                                              rand_hour(arrDep_dict['mean_arr_work']+var, tot_bldg) ) )
        cur_arr_dep+=1
        
        # If charging at work is considered in the simulation i.e. chrg_at_work=True 
        if chrg_at_work : 
            # For curent day, Make sure  arrival time < to previous departure time 
            while (dep_dict_f[cur_arr_dep]<arr_dict_f[cur_arr_dep+1]).sum() < tot_bldg : 
                dep_dict_f[cur_arr_dep] = np.minimum(arrDep_dict['max_dep_work'] + var,
                                                     np.maximum( arrDep_dict['min_dep_work'] + var,
                                                                rand_hour(arrDep_dict['mean_dep_work'] + var ,tot_bldg) )) 
                arr_dict_f[cur_arr_dep+1] = np.minimum(arrDep_dict['max_arr'] + var, 
                                                       np.maximum(arrDep_dict['min_arr'] + var,
                                                                  rand_hour(arrDep_dict['mean_arr']+var, tot_bldg)))
            cur_arr_dep+=1

    # Initialize last interval
    dep_dict_f[nb_arr_dep] = np.maximum(arrDep_dict['min_dep']+nb_sim_days*probVar.fullDay_min/probVar.Δt ,
                                        rand_hour(arrDep_dict['mean_dep'] + nb_sim_days*probVar.fullDay_min/probVar.Δt, tot_bldg))
    
    
#     #make sure all the the arrival and departure time are multiple of Δt
#     for i in range(1, nb_arr_dep+1):
#         # (Δt - depart1%Δt) gives the distance to become a multiple of Δt | ad the found number to the
#         dep_dict[i] = Δt-dep_dict[i]%Δt + dep_dict[i]
#         arr_dict[i] = Δt-arr_dict[i]%Δt + arr_dict[i]

    arr_dict_f[1] = np.zeros((tot_bldg)) # Reset first arrival instant to 0 
    
    return arr_dict_f, dep_dict_f





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def init_evDataParams(arr_dict, dep_dict, soc_arr_dict, soh_arr1, evs_data_df, nb_arr_dep ):
    """ 
    Initialize Ev parameters for all EV described in ''evs_data_df''
    
    Inputs:
    -------
    arr_dict: Dict
        Arrival instant dictionnary for all EVs i.e.
        first output of brp.init_arrDep_instants(args)
    dep_dict: Dict
        Departure instant dictionnary for all EVs i.e.
        second output of brp.init_arrDep_instants(args)
    soc_arr_dict: Dict 
        Dictinnary of Soc at each arrival instants
    soh_arr1 : 1D array
        State of health of EV storage at the beginning of simulations. 
    evs_data_df : Dataframe
        Dataframe describing the EV number and the associated bus (building).
        
        
    NB: The first 4 imputs are actually associated not directly with the Evs but 
    with the buildings in the networks. As for the fifth input evs_data_df,  
    Its lenght is the actual number of activ EVs to consider in the simulation. 
    
    
    Outputs:
    --------
    ev_data_dict : Dict
        A dictionary of EVs parameters for all EVs conected to each load bus
        
    """
    
    ev_data_dict = {} # Dict to store ev data parameters
    
    for i in range(len(evs_data_df)): # For each active bus with an Ev connected

        cur_Ev = {} #dict to store current ev data

        for cur_arr_dep_ind in range(1, nb_arr_dep+1):# For each arrival and departure interval
            # Add values to dictionary
            cur_Ev.update({'soc_arr' + str(cur_arr_dep_ind):  soc_arr_dict[cur_arr_dep_ind][i], 
                            't_arr'  + str(cur_arr_dep_ind):  arr_dict[cur_arr_dep_ind][i],
                            't_dep'  + str(cur_arr_dep_ind):  dep_dict[cur_arr_dep_ind][i]})

        cur_Ev.update({'soh_arr1':  soh_arr1[i],
                       'soc_f':     probVar.soc_min_dep,
                       'nb_arr_dep':nb_arr_dep })

        #upate ev data
        ev_data_dict.update({evs_data_df['Bus'][i]: cur_Ev } )
        
    return ev_data_dict





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def printArrDep_time(evs_data_params, tarr ='t_arr1',  tdep ='t_dep1', nbr_bus = 1):
    """Print arrival and departure time associated with each interval considered in the simulation. 
    
    Inputs
    ------
    evs_data_prams : Dict
        A dictionary of EVs parameters for each EV conected to each load bus. 
        Output of the function << brp.init_evDataParams(args) >>
    tarr: String, Default 't_arr1'
        String indicating the current arrival time
    tdep: String, Default 't_dep1'
        String indicating the current departure time
    nbr_bus: Int, Default 1'
        Number of bus for which to print arrival and departure interval
    
    """
    
    # Get the list of bus with EV in the network
    bus_list = list(evs_data_params.keys())
    
    print('  \t \t      '+tarr +',     '+tdep)
    print('   \t \t    ----------------------')
    for ev_bus in bus_list[:nbr_bus]: 
        print('Ev on bus:', ev_bus,': ', list(map(periodToHod, np.array([evs_data_params[ev_bus][tarr], 
                                                                         evs_data_params[ev_bus][tdep]], int))) )


        
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    


def compute_Pplan(p_pv_plan, p_load_plan, p_ev_plan ):
    """Compute Pplan for the simulation. 
    
    Inputs
    ------
    p_pv_plan : Dict
        A dictionary of pv_production  for each bus (building) in the network
    p_load_plan : Dict
        A dictionary of load demand  for each bus (building) in the network
    p_ev_plan: Dict
        A dictionary of ev demand  for each active bus (building) i.e. each bus 
        that has an ev connected to it 
        
    
    Output
    ------
    p_plan : List
        P_Plan value for each imbalance period for the whole simulation
    
    
    """
    # Planned BRP power for each settlement period
    p_plan=[]
    
    day2_imb = int(probVar.day2_min/probVar.Δt) # Starting imbalance number of day 2

    
    ΔT = probVar.ΔT
    
    # Compute the total number of imbalance period
    nbr_imb_periods = len(p_ev_plan[list(p_ev_plan.keys())[0]])//ΔT

    for T in range(0,nbr_imb_periods):
        plan_sum = 0
        for cur_bus in p_pv_plan:
            var_pv_plan = np.sum((p_pv_plan[cur_bus][(T*ΔT +day2_imb) :(T+1)*ΔT+day2_imb]))
            var_pload_plan = np.sum(np.array(p_load_plan[cur_bus][(T*ΔT +day2_imb) :(T+1)*ΔT+day2_imb ])*(1+probVar.err_load)) 
            plan_sum = plan_sum + var_pv_plan - var_pload_plan

            if cur_bus in p_ev_plan.keys():# If the current bus 'cur_bus' has an ev connected to it 
                plan_sum = plan_sum - np.sum(np.array(p_ev_plan[cur_bus][T*ΔT :(T+1)*ΔT ])*(1+probVar.err_ev))

        p_plan.append((1/1000)*plan_sum*probVar.s_base/ΔT ) # in kw and covers only Three days starting for the simulation day
        
        
    return p_plan




#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def interval_finder(x, evData_curBus_dict):
    """Find the interval within wich the input x is located
    
    Inputs:
    -------
    x : Int
        Some instant of the problem
    evData_curBus_dict: dict 
        Dictionary that consists of a the data related to time, soc, and soh of
        at arrival and departure of the ev connected to the current bus
        
    Outputs: 
    --------
    (a,b,c) : tuple int 
        a    ==> i  if x in [t_arr_i, t_dep_i[ the current number of arrival and departure
                    i.e. Ev on current bus is currently plugged-in
             ==> 0  Otherwise i.e. ev is unplugged 
       (b,c) ==> (t_arr_i, t_dep_i) if x in [t_arr_i, t_dep_i[
             ==> (0, 0)   otherwise
    """
    #for each time where the Ev is connected
    for i in range(1, int(evData_curBus_dict['nb_arr_dep'])+1):
        
        arrival = 't_arr' + str(i)   # String to index arrival time
        departure = 't_dep' + str(i) # String to index departure time
        
        # if current instant is  in [t_arr_i, t_dep_i[
        if x in range(int(evData_curBus_dict[arrival]), int(evData_curBus_dict[departure])):
            out_val = (i, int(evData_curBus_dict[arrival]), int(evData_curBus_dict[departure])) # Set the output values
            break 
        else:
            out_val = (0,0,0)
            
    
    return out_val





def dumb_en_mis_at(cur_k, mpcHrz_in_min, emis_dumb_df_f ):
    # TODO : Write a function to produce the energy mismatch dataframe (emis_dumb_df_f) based on the 
    # ΔT 
    """Extract the dumb energy mismatch over the 10 previous days 
    
    Inputs:
    -------
    cur_k : Int
         Current instant of the simulation
    mpcHrz_in_min : Int 
        MPC horizon in minutes
    emis_dumb_df : panda dataframe
        Energy mismatch of 10 previous days using the dumb strategy at a freq of 15mn
     
    """
    
    
    # cur_ress_per: Current resample period 
    # emis_dumb_df_f : dumb energy mismatch dataframe for each imbalance period over the whole 10previous days 
    # horiz= Horizon in minutes
    
    time_step = cur_k // probVar.ΔT              # in one imbalance period there are ΔT periods
    time_step2 = time_step + (mpcHrz_in_min//15) # Get the instant of the last imbalance period 
    
    
    df_to_work = emis_dumb_df_f.iloc[time_step:] # Extract the part of the dataframe one wants to work with, 
                                                   # i.e from the current imbalance period to the future.

    var_int = df_to_work.index[0]           # Get index of the initial instant
    var_init = str(var_int)[0:8]            # Extract year and month '2019-07-' 
    var_end = str(var_int)[-8:]             # Extract hour '01:15:00' # 
    curr_day = var_int.day                  # Extract the day 
    
    en_miss = []                           # list to store energy mismatch for each day
    
    # for each day in the previous days
    for elm in range(curr_day, curr_day+11): # 11 day and plus in dataframe
        hour_start = var_init + str(elm)+ ' ' + var_end  # current imbalence period timeframe 
        hour_end = str(pd.date_range(hour_start, periods=2, freq=str(mpcHrz_in_min)+'T')[1]) #create 
                                          #the last instant timeframe covered by the horizon 
        
        # Extract period between the first and last imbalance period covered by horrizon
        bin_condition = (df_to_work.index >= hour_start) & (df_to_work.index < hour_end)
        emis_t1_t2 = df_to_work.loc[bin_condition]
        
        if emis_t1_t2.shape[0] == (mpcHrz_in_min//15):
            en_miss.append(emis_t1_t2.sum())# sum up all elements in emis_t1_t2

    if len (en_miss) != 0 : return np.array(en_miss).mean()
    else : return 0
    
    
    
    

#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    
    
def compute_DumbSoc (p_ev_plan, ev_data_dict, t_sim ):
    """
    Compute the dynamic of soc  (all EV in the network) using the dumb strategy 
    which charges at constant rate As soon as  the EV is plugged in
    
    Inputs:
    -------
    p_ev_plan: Dict
        A dictionary of ev demand  for each active bus (building) i.e. each bus 
        that has an ev connected to it/
    ev_data_dict: Dict
        Dictionnoary containnig all the parameters of the Evs connected 
        to the network;  Output of function << brp.init_evDataParams(args) >>
    t_sim: Int
        Total simulation time
        
        
    Output:
    -------
    soc_prob_dumb: Dict
        Dynamic of Soc of all Ev in the network. Defined over the problem horizon. 
        

    """
    
    # Import variables to use 
    from probVar import eff_chrg,s_base, e_bat, e_tp, soc_min, soc_max, soc_min_dep
    

    soc_prob_dumb = {} # Dictionary to store the soc for the dumb strategy
    
    inter_var = (probVar.Δt*eff_chrg*s_base)/(60*e_bat)# An itermediate variable

    for cur_bus in probVar.evs_data_df['Bus'].values: # For each EV in the network 
        var_len = len(p_ev_plan[cur_bus])
        var_soc = soc_min*np.ones((1,var_len))

        for cur_ins in range(1,ev_data_dict[cur_bus]['nb_arr_dep']+1):
            arr1 = int(ev_data_dict[cur_bus]['t_arr'+str(cur_ins)])
            dep1 = int(ev_data_dict[cur_bus]['t_dep'+str(cur_ins)])

            var_soc[0,arr1] = ev_data_dict[cur_bus]['soc_arr'+str(cur_ins)]
            var_soc[0,arr1+1:dep1+1] = np.cumsum(p_ev_plan[cur_bus][arr1:dep1]*inter_var) + ev_data_dict[cur_bus]['soc_arr'+str(cur_ins)]

        soc_prob_dumb.update({cur_bus: var_soc[0,:t_sim+1]})
        
    return soc_prob_dumb





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def update_ppv_pred(cur_k, lb_curDay, curDay_frcst_per, per_back_windows, prob_predValues, per_daylight, input_data, cur_regime):
    """
    Update The predicted  irradiance
    
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
        Number of period covers by the backward 
        looking windows to compute the MSE
    curDay_predValues: 1D float array (1*288)
        Predicted vector of the current day
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
                the function << dayType_dataPred(args) >>     
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
    
    daily_hor = 1440//5 # Daily horizon to cover.
    
    upd_cur_k = cur_k-lb_curDay # Updated current k so that the updated cur_k is always in [0, 288], Δt = 5 mn

    if cur_k in curDay_frcst_per :  # If updating instant === Recompute MSE and update given the regime producing the lowest MSE
        err_list = []
        for day_type in range(0,4): 
            # Compute mean square error between actual irradiance and the current predicted 
            datta = zeroing(curDay_data, per_start_daylight, per_end_daylight)['GHI'] - curDay_data_dict[day_type] # Extract Day 
            err = (datta[upd_cur_k-per_back_windows: upd_cur_k]**2).sum()/per_back_windows # Compute Mean squared error 
                                                                                           # usinng the back looking windows lenght
            err_list.append( err ) # Append error to a list 

        # Get index of the element that has produced lowest MSQ
        cur_regime = np.array(err_list).argmin()
        #update predictec values for the future using the regime that has produced the lowest MSQ
        prob_predValues[cur_k: cur_k+ daily_hor ] = prop_curDayToHor(upd_cur_k, daily_hor, curDay_data_dict[cur_regime])
        
    else: # Otherwise Keep using the same regime as previous step
        
        prob_predValues[cur_k: cur_k+ daily_hor ] = prop_curDayToHor(upd_cur_k, daily_hor, curDay_data_dict[cur_regime])
  
                        
    return prob_predValues, cur_regime





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def prop_curDayToHor(upd_cur_k, daily_hor, curDay_data_dict_indx ):# Propagate current day to fill the horizon of one day
    """ 
    Update the predicted GHI (Global Horizontal Iradiance)  produced by the regime with the lowest MSQ over 
    a period of 24h (maximum prediction period of MPC)  
    
    
    Inputs:
    -------
    upd_cur_k: Int
        Updated current k so that the updated cur_k is always in [0, daily_hor], Δt = 5 mn
    daily_hor: Int
        Daily total horizon
    curDay_data_dict_indx: float Array (1*daily_hor)
        Irradiance prediction of the regime that has produced the lowest MSE
        
        
    Output:
        pred_val_daily_hor: float Array (1*daily_hor)
        
    """
    
    pred_val_daily_hor = np.zeros((daily_hor)) # create a vatiable to store the predicted value of the current
                                # regime of P_PV iradiance over a period of one day
        
    remain_hor = daily_hor - upd_cur_k # Remaining Horizon to cover  i.e if upd_cur_k = 200,
    # ==> remain_hor = 288 - 200 = 88  
    
#     fill the predicted value over the horizon of one day
    pred_val_daily_hor[:remain_hor] = curDay_data_dict_indx[upd_cur_k:]  
    pred_val_daily_hor[remain_hor:] = curDay_data_dict_indx[upd_cur_k:]  
    
    
    return pred_val_daily_hor
    
    