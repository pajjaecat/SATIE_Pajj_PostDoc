"""

Definig function to use in the MPC problem

"""

# My own module
import probVar, brp, pickle, regSwitchSolarPred as rssp

# get numpy,  pandas and cvxpy  from probVar
np = probVar.np
pd = probVar.pd
cp = probVar.cp




def createOptim_var (mpcHrz, totBus_network):
    """ Define the optimisation variables to use over the MPC problem horizon
    NB: Add new variables to this function 
    
    Inputs:
    ------
    mpcHrz: Int
        Number of periods covered by the MPC horizon
    totBus_network : Int
        Total number of Bus in the network
        
    Output:
    ------
        All the created variables as tuple over the mpc horizon such that
            (0) Active power related Variables
                ==> p_var = (p_ev_chrg, del_p_ev, p_pv_hor_dict, p_load_hor_dict, p_bus, p_gen)
            (1) Ev Storage related variables
                ==> sto_var = (soc, soc_relax, soh)
            (2) Network related variables
                ==> line_var = (theta, voltage)
             
    """

    
    #creating dictionary for all variables
    #Variables for Active power decisions
    p_ev_chrg = {}      # EV Active Charging power,
    del_p_ev = {}       #Difference in planned EV active variable

    #Storage variables
    soc = {} #State of charge 
    soc_relax = {} # Departure soc relaxation variable
    soh = {} # #State of health

    # Variables for p_pv and p_load over the MPC horizon
    p_pv_hor_dict , p_load_hor_dict = {}, {}
    #q_pv_hor_dict ,q_load_hor_dict = {}, {}
    
    # Line variables 
    theta = cp.Variable((totBus_network+1, mpcHrz) )
    voltage = cp.Variable((totBus_network+1, mpcHrz))
    p_bus = cp.Variable((totBus_network+1, mpcHrz))
    p_gen = cp.Variable((1,mpcHrz))
    
    for cur_bus in probVar.evs_data_df['Bus'].values: # for each bus with ev
        #active
        p_ev_chrg.update({cur_bus: (cp.Variable((1,mpcHrz), nonneg=True)) })
        del_p_ev.update({cur_bus: (cp.Variable((1,mpcHrz) ) ) })

        # Storage variables and soh
        soc.update({cur_bus: (cp.Variable((1,mpcHrz+1), nonneg=True)) })
        soc_relax.update({cur_bus: (cp.Variable((1), nonneg=True)) })
        soh.update({cur_bus: (cp.Variable((1,mpcHrz+1), nonneg=True)) })
    
    p_var = (p_ev_chrg, del_p_ev, p_pv_hor_dict, p_load_hor_dict, p_bus, p_gen)
    sto_var = (soc, soc_relax, soh)
    line_var = (theta, voltage )

    return p_var, sto_var,line_var 
    



    
#________________________________________________________________________________________________________________  
#----------------------------------------------------------------------------------------------------------------  
#________________________________________________________________________________________________________________ 
    
def fill_busConstraints(cur_k, cur_bus, mpcHrz, 
                        p_pv_dict, p_load_dict, 
                        p_pv_hor_dict, p_load_hor_dict,
                        bus_cons, 
                        p_bus, p_ev_chrg,
                        pv_pred_type='known', 
                        pred_input=0):
    """
    Fill the bus constraints related to p_pv, p_load, p_ev at the current instant (cur_k) for 
    the current bus (cur_bus) over the MPC horizon horizon
    
    Inputs:
    ------
    cur_k: Int
        Current simulation step k 
    cur_bus: Int
        The active bus number
    mpcHrz: Int
        Number of periods covered by the MPC horizon
    p_pv_dict: Dict
        Dictionnary of real pv production for each active bus connected to a PV panel
    p_load_dict: Dict
        Dictionnary of real load demand for each active bus in the network
    p_pv_hor_dict: Dict
        Dictionnary of pv production defined over the MPC horizon. Updating 
        it in the function, add the P_pv of the current bus to the said dict.
    p_load_hor_dict: Dict
        Dictionnary of load demand defined over the MPC horizon. Updating 
        it in the function, add the Load of the current bus to the said dict.
    bus_cons: List
        Bus constraints list 
    p_bus: cvxpy variable 
        Optimization variable 
    p_ev_chrg: cvxpy variable 
        Charging variable value
    pv_pred_type: String, default 'known'
        Pv prediction type to use in mpc's horizon
        'known'   ==> Future is perfectly known
        'pers'    ==> Future is based on persistence model of previous day
        'rgSwitch'==> Future is based on the regime switching prediction model
    pred_input: Tuple
        All the inputs to be used when using the model swiching PV as prediction
        See the function << pred_Irradiance (args) >> for the said inputs 

        
        
    Output: 
    -------
        Updated bus_cons, p_pv_hor_dict and  p_load_hor_dict
    
    """
    
    if cur_bus in p_pv_dict : # If the current bus has some Pv connected to a building  
        #P_load an P_pv depend on the active  bus
        p_load_hor = np.zeros((mpcHrz))
    #             q_load_hor = np.zeros((hor))

        curr_ins = (probVar.day2_min//probVar.Δt) + cur_k       # Define current instant in relation to the starting day 2

        # Defined P_Pv horizon
        p_pv_hor = def_ppv_hrz(cur_k, mpcHrz, curr_ins, p_pv_dict[cur_bus], pv_pred_type, pred_input  )

        p_load_hor = p_load_dict[cur_bus][curr_ins:curr_ins+mpcHrz]  # p_load_hor is known and is the real p_load of day 2
    #    q_load_hor = q_load[i][curr_ins:curr_ins+hor]  # The remaining elements are given by the value of the previous day

        # add pv and load over the horizon to 
        p_pv_hor_dict.update({cur_bus: p_pv_hor})
        #q_pv_hor_dict.update({i: #q_pv_hor})        
        p_load_hor_dict.update({cur_bus: p_load_hor})
    #             q_load_hor_dict.update({i: q_load_hor})

        if cur_bus in probVar.evs_data_df['Bus'].values: #The current bus has an ev associated
            bus_cons.append(p_bus[cur_bus] == p_pv_hor_dict[cur_bus] - p_load_hor_dict[cur_bus]  - p_ev_chrg[cur_bus][0,:])
            #bus_constraints.append(q_bus[i] == q_pv_hor_dict[i] - q_load_hor_dict[i]  - q_ev_chrg[i][0,:])
        else: #The current bus has no ev associated
            bus_cons.append(p_bus[cur_bus] == p_pv_hor_dict[cur_bus]  - p_load_hor_dict[cur_bus])
            #bus_constraints.append(q_bus[i] == q_pv_hor_dict[i]  - q_load_hor_dict[i])

    else: # The current bus is not connected to a building
        bus_cons.append(p_bus[cur_bus] == 0) 
        #bus_constraints.append(q_bus[i] == 0)
        
    return bus_cons, p_pv_hor_dict, p_load_hor_dict
            
            
    

    
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________  

def def_ppv_hrz(cur_k, mpcHrz, curr_ins, p_pv_curBus, pv_pred_type='known', pred_input=0):
    """ Define the P_pv to use over the MPC problem horizon
    
    Inputs:
    ------
    cur_k: Int
        Current simulation step k 
    mpcHrz: Int
        Number of periods covered by the MPC horizon
    curr_ins: Int
        Current instant defined in relation to the starting of day 2
    p_pv_curBus: 1D array
        pv production for the current bus
    pv_pred_type: String, default 'known'
        Pv prediction type to use in mpc's horizon
        'known'   ==> Future is perfectly known
        'pers'    ==> Future is based on persistence model of previous day
        'rgSwitch'==> Future is based on the regime switching prediction model
    pred_input: Tuple
        All the inputs to be used when using the model swiching PV as prediction
        See the function << pred_Irradiance (args) >> for the said inputs 
        
    Output:
    ------
    p_pv_hrz : 1D array 
        P_pv defined over the MPC horizon
    """

    
    p_pv_hrz = np.zeros((mpcHrz))
    #q_pv_mpcHrz = np.zeros((mpcHrz))

    p_pv_hrz[0] = p_pv_curBus[curr_ins]  # The first element of p_pv_hor is the real p_pv of day 2
    
    # Choose the predicted value of future instants given the prediction type entered by the user 
    if pv_pred_type == 'known':  #------ Future known
        p_pv_hrz[1:] = p_pv_curBus[curr_ins+1:curr_ins+mpcHrz] #The remaining elements are given by real value of P_pv
#         print(p_pv_curBus[curr_ins+1:curr_ins+mpcHrz])

    elif pv_pred_type == 'pers' :#------ Persistence model of previous day
        p_pv_hrz[1:] = p_pv_curBus[cur_k+1:cur_k+mpcHrz]    # The remaining elements are given by the value of P_pv the 
                                                            #  previous day

    elif pv_pred_type == 'rgSwitch' :#------Regime switching model
        pred_output = pred_Irradiance(cur_k, *pred_input)
        p_pv_hrz[1:] = pred_output.values[0, 1:mpcHrz]*1000/probVar.s_base

    else: 
        raise ValueError('Wrong pv_pred_type, Must be ''known'', ''pers'', or ''rgSwitch'' ')
        
    return p_pv_hrz





#________________________________________________________________________________________________________________  
#----------------------------------------------------------------------------------------------------------------  
#________________________________________________________________________________________________________________ 

def fill_lineConstraints(line_cons, B, G, 
                         theta, voltage, p_bus, 
                         pbus_bound=1., conLinPower_bound=2.):
    """
    Fill the line  constraints
    
    Inputs:
    ------
    line_cons: List
        line constraints list 
    p_bus: cvxpy variable 
        Optimization variable 
    ...  
    pbus_bound: float, default=1
        Bound of the maximum power in each bus 
    conLinPower_bound: Float, default= 2 # congested Line Power bound
        Bound of the maximum power to reach congestion
        
    Output: 
    -------
        Updated line constraints list
    
    """
    
    line_cons.append(p_bus == (B@theta + G@voltage))
    #line_constraints.append(q_bus == (B@voltage - G@theta)) 
    
    line_cons.append((p_bus) <=  pbus_bound) #
    line_cons.append((p_bus) >= -pbus_bound) #


    #Constraint to model line congestions
    congested_line_power = cp.sum([p_bus[a] for a in range(39,206)],axis=0)  #Subdistric 1 is defined between Bus 39 and 207 
    line_cons.append(congested_line_power <=  conLinPower_bound)  # Perhaps remove these constraints later on
    line_cons.append(congested_line_power >= -conLinPower_bound) 

        
    return line_cons
            
            
        
        
    
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def fill_Part_vehConstraints(cur_k, kk, mpcHrz, veh_cons,
                             p_var_curBus, soc_var_curBus, 
                             ev_data_curBus, inter_type): 
    """
    Fill the vehicle constraints dependant on wether the current instant and the 
    last instant, repectively  cur_k and cur_k+mpcHrz are on the same interval
    
    Inputs:
    ------
    cur_k: Int
        Current simulation step k 
    kk: tuple
        output of brp.interval_finder(args) for the instant k
    mpcHrz: Int
        Number of periods covered by the MPC horizon
    veh_cons: List
        Vehicle constraints list 
    p_var_curBus: Tuple
        Variables associated with P_ev for the current bus
        (0) pev_plan_curBus: 1D float array 
                Planned Ev  power for the vehicle connected to the current bus, 
                defined over the whole problem horizon
        (1) pev_chrg_curBus: 1D cvxpy variable
                Decision variable, P_ev_charge, Defined over MPC horizon
        (2) del_pev_curBus: 1D cvxpy variable
                Decision variable, del_pev_curBus, Defined over MPC horizon
        (3) pev_charBin_curBus: 1D binary Array
                Equals 1 at the instants where the EV on the current bus is 
                plugged-in and 0 otherwise
    soc_var_curBus: Tuple
        Variables associated with the soc for the ev on the current bus
        (0) soc_curBus: 1D cvxpy variable
                Decision variable, soc_curBus, Defined over MPC horizon
        (1) soc_relax_curBus: cvxpy variable
                Decision variable, soc, relaxaion variable 
    ev_data_curBus: Dict
        Dictionnoary containnig all the parameters of the current Ev
        connected to the bus 
    inter_type: String
        Whether the current instant and the end of the horizon are on the same interval
        'same'   ==> Same interval
        'diff'   ==> different interval  
        
        
    Output: 
    -------
    Depending on the input <inter_type>
        <inter_type> == 'same'
            ==> veh_cons Updated list of vehicle constraints
        <inter_type> == 'diff'
        ==> (veh_cons,index_ev_con)  Updated list of vehicle constraints and 
            index of instants where the vehicle is plugged in 

    """
    
    
    # unpack inputs
    pev_plan_curBus, pev_chrg_curBus, del_pev_curBus, pev_charBin_curBus  = p_var_curBus
    soc_curBus, soc_relax_curBus = soc_var_curBus
    
    
    if inter_type == 'same' :################################  Same interval ####################################################

        if (kk[0] != 0): #Ev is connected to the grid ==> Interval 1,2,3,....

            t_ev_arr = kk[1] # Ev_ arrival time 
            t_ev_dep = kk[2] # Ev_ departure time

            veh_cons.append(pev_chrg_curBus[0, 0:mpcHrz] <= (probVar.p_max*1000/probVar.s_base))
            veh_cons.append((pev_chrg_curBus[0, 0:mpcHrz] ) == (pev_plan_curBus[cur_k:cur_k+mpcHrz] + del_pev_curBus[0, 0:mpcHrz]))

            if (t_ev_dep <= cur_k+mpcHrz): #  if Ev  departure is covered by the current horizon relax departure soc constraints
                veh_cons.append(soc_curBus[0, kk[2]-cur_k] >= ev_data_curBus['soc_f'] -  soc_relax_curBus)
    #                     print('bus', i, ' relax, same interval ')
    #                     veh_cons.append(soc_curBus[0, kk[2]-k] >= ev_data[cur_bus]['soc_f'] )


    #                 nb_inst = t_ev_dep - (k+mpcHrz) # Compute the number of instants separating the MPC horizon's end to the Ev_departure
    #                 soc_hor = max(ev_data[cur_bus]['soc_f'] - nb_inst*Δt*(p_ev_char_max -2e-4),soc_min) # Compute the minimum soc at k+hor 
    #                 veh_cons.append(soc_curBus[0, mpcHrz:mpcHrz+1] >= soc_hor ) # Propagate backwards the departure soc to the MPC 
    #                                                                              # horizon's end 
            # Compute the SOC dynamic          
            veh_cons.append(soc_curBus[0,1:] == soc_curBus[0][0:mpcHrz] + 
                            ((probVar.Δt*pev_chrg_curBus[0,0:]*probVar.eff_chrg)*probVar.s_base/60)/probVar.e_bat )

        
        else: # Ev is disconnected from the grid ==> intervals 0 ----------------------------
            veh_cons.append(pev_chrg_curBus[0,0] == 0) 

    #             if (kk[0]==3) & ((day_2_imba-k)< mpcHrz):# enforce ending of day one soc to be the same for all different horizon                 
    #                 var = ((Δt*pev_plan_curBus[:day_2_imba]*eff_chrg)*s_base/60)/e_bat # Charging power at each instant over day of simulation
    #                 var2 = np.sum(var[int(ev_data[cur_bus]['t_arr3']):])# Compute sum of charging power from the arrival the second time 
    #                                                               # to the end day of simulation
    #                 soc_var = np.round(ev_data[cur_bus]['soc_arr3'] + var2, decimals=1)
    #                 veh_cons.append(soc_curBus[0, day_2_imba-k] == soc_var )# Set the constraints
        
        return veh_cons
    
    
    elif inter_type=='diff': #############################  Different interval ##################################################""---
       
        if kk[0] == 0 : # Ev is not connected at the initial instant k --------------------------------------

             # Over the MPC horizon, find index of instants where the vehicle is plugged in 
            index_ev_con = pev_charBin_curBus[cur_k:cur_k+mpcHrz] == 1 

            index_ev_con[:] = 0 # Given that when the ev is not connected we supposed that 
            # no information about its next arrival time is given, set all the connected  
            # instant value to zero
            veh_cons.append(pev_chrg_curBus[0,0] == 0) 
        
        else : # Ev is connected at the initial instant-------------------------------------------------

            # Over the MPC horizon, find index of instants where the vehicle is plugged 
            index_ev_con = pev_charBin_curBus[cur_k:cur_k+mpcHrz] == 1 

            # Given that the focus is only on the current interval i (t_arr_i, t_dep_i)
            # set the p_ev outside the interval to 0
            # kk[2] = t_dep_i
            index_ev_con[int(kk[2] -cur_k):] = 0

            if sum(index_ev_con)> 0: # If there at least one instant where the vehicle is connected 
                veh_cons.append(pev_chrg_curBus[0,index_ev_con] <= (probVar.p_max*1000/probVar.s_base))        
                var_int2 = np.array(pev_plan_curBus[cur_k:cur_k+mpcHrz])

                #vehicle_constraints.append((q_ev_chrg[cur_bus][0, index_ev_con] ) == (var_int1[index_ev_con] + del_q_ev[cur_bus][0, index_ev_con]))
                veh_cons.append(pev_chrg_curBus[0, index_ev_con]  == (var_int2[index_ev_con] + del_pev_curBus[0, index_ev_con]) )

            # Set soc_f at the end of the current interval i.e., the departure time of the current instant t_dept_i
            veh_cons.append(soc_curBus[0, kk[2]-cur_k] >= ev_data_curBus['soc_f'] - soc_relax_curBus)  # soc at the end of interval 1
#                 print('bus', i, ' relax, diff interval ')

        return veh_cons, index_ev_con

    else: 
        raise ValueError('The interval type must be either ''same'' or '' diff'' ')
        
        

        

#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    
                
def fill_vehConstraints(cur_k, mpcHrz, veh_cons, p_var_dict, soc_var_dict, soh_var_dict,  ev_data_dict): 
    """
    Fill all the vehicle constraints
    
    Inputs:
    ------
    cur_k: Int
        Current simulation step k 
    mpcHrz: Int
        Number of periods covered by the MPC horizon
    veh_cons: List
        Vehicle constraints list 
    p_var_dict: Tuple
        Variables associated with P_ev for the current bus
        (0) pev_plan_dict: Dict of 1D float array 
                Planned Ev power for all the vehicles connected to the network, 
                defined over the whole problem horizon
        (1) pev_chrg_dict: Dict of 1D cvxpy arraya
                Decision variable, P_ev_charg, Defined over MPC horizon
                for all the Ev connected to a bus
        (2) del_pev_dict: Dict of 1D cvxpy arraya
                Decision variable, del_pev_dict, Defined over MPC horizon
                for all the Ev connected to a bus
        (3) pev_charBin_dict: Dict of 1D binary Array
                Binary array indicationg 1 at the instants where the EV is plugged-in 
                and 0 otherwise, defined over the problem Horizon
    soc_var_dict: Tuple
        Variables associated with the soc for the ev on the current bus
        (0) soc_prob_dict: Dict 1D float array
                Optimal value of Decision variable soc, defined over the problem horizon
        (1) soc_hor_dict: Dict of 1D cvxpy variable
                Decision variable, soc of all EV in the network, Defined over MPC horizon
        (2) soc_relax_dict: Dict cvxpy variable
                Decision variable, relaxaion variable of soc
    soh_var_dict: Tuple
        Variables associated with the soc for the ev on the current bus
        (1) soh_prob_dict: Dict 1D float array
                Optimal value of Decision variable soh, defined over the problem horizon
        (0) soh_hor_dict: Dict of 1D cvxpy array
                Decision variable, soh of all EV in the network, Defined over MPC horizon
    ev_data_dict: Dict
        Dictionnoary containnig all the parameters of the Evs connected 
        to the network;  Output of function <<brp.init_evDataParams(args)>>
        
        
    Output: 
    -------
        veh_cons: List
            List of vehicle constaints 
        relax_list: List (cvxpy variable)   
            List of relaxation variable for each EV in the network
    
    """

    # Import variables to use 
    from probVar import eff_chrg,s_base, e_bat, e_tp, soc_min, soc_max, soc_min_dep
    
    relax_list = [] #vehicle constraints list

    
    # unpack inputs
    pev_plan_dict, pev_chrg_dict, del_pev_dict, pev_charBin_dict  = p_var_dict
    soc_prob_dict, soc_hor_dict, soc_relax_dict  = soc_var_dict
    soh_prob_dict, soh_hor_dict, = soh_var_dict
    
    for cur_bus in probVar.evs_data_df['Bus'].values: # For each ev on the network 
        
        # find in wich interval the current instant and the last instant(k+hor) are located 
        kk = brp.interval_finder(cur_k, ev_data_dict[cur_bus])
        kk_hor = brp.interval_finder(cur_k+mpcHrz, ev_data_dict[cur_bus])
        
        # Different treatment depending on the location of k and k+hor
        #________________________________________________    Same interval   ____________________________________________________________
        if (kk[0]==kk_hor[0]): 
            veh_cons = fill_Part_vehConstraints(cur_k, kk, mpcHrz, veh_cons,
                                                    (pev_plan_dict[cur_bus], pev_chrg_dict[cur_bus], del_pev_dict[cur_bus], pev_charBin_dict[cur_bus]),
                                                    (soc_hor_dict[cur_bus], soc_relax_dict[cur_bus]), ev_data_dict[cur_bus], 'same' ) 
            
        #______________________________________________    Different interval    _______________________________________________________
        else: 
            veh_cons, index_ev_con = fill_Part_vehConstraints(cur_k, kk, mpcHrz, veh_cons,
                                                                  (pev_plan_dict[cur_bus], pev_chrg_dict[cur_bus], del_pev_dict[cur_bus], pev_charBin_dict[cur_bus]),
                                                                  (soc_hor_dict[cur_bus], soc_relax_dict[cur_bus]), ev_data_dict[cur_bus], 'diff' ) 

            # compute dynamic for only instants where the ev is connected to the grid
            if sum(index_ev_con)> 0:
                veh_cons.append(soc_hor_dict[cur_bus][0,1:][index_ev_con] == soc_hor_dict[cur_bus][0][0:mpcHrz][index_ev_con] +
                                           ((probVar.Δt*pev_chrg_dict[cur_bus][0,0:mpcHrz][index_ev_con]*eff_chrg)*s_base/60)/e_bat )
    
        # add the remaining vehicle constraints    
        # add SOH constraints
        veh_cons.append(soh_hor_dict[cur_bus][0,0:1] == soh_prob_dict[cur_bus][0,cur_k] )
        veh_cons.append(soh_hor_dict[cur_bus][0,1:] == soh_hor_dict[cur_bus][0,:mpcHrz] - (probVar.Δt*pev_chrg_dict[cur_bus][0,0:]*s_base/60)/(0.2*e_tp) ) 
        veh_cons.append(soh_hor_dict[cur_bus][0,1:] >= 0)            
        # add  Soc constraints
        veh_cons.append(soc_hor_dict[cur_bus][0,0:1] == soc_prob_dict[cur_bus][0,cur_k])
        veh_cons.append( soc_min <= soc_hor_dict[cur_bus][0,1:] )
        veh_cons.append( soc_hor_dict[cur_bus][0,1:] <= soc_max )
        
        # Stored the normalized relaxation variable for the current ev (add to the list)
        # max(0, (soc_min - soc_dep)/soc_min) 
#         relax_list.append((soc_relax[cur_bus]/soc_min_dep)**2)
        relax_list.append( (soc_relax_dict[cur_bus]/(soc_min_dep-soc_min)) )
    
    
    return veh_cons, relax_list

 
        
        

#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def imbalancePer_Emis(cur_T, cur_k, mpcHrz, totBus_network, p_plan_curT, p_pv_hor_dict, p_load_hor_dict, p_ev_chrg_f):
    
    """Computes the energy mismatch of the current imbalance period cur_T
    
    Inputs:
    -------
    cur_T : Int 
        Current imbalance period
    cur_k : Int 
         Current instant of the simulation
    mpcHrz : Int
        MPC horizon
    totBus_network : Int
        Total number of Bus in the network
    p_plan_curT: Float 
        Engagement, i.e. planned power for the current imbalance period
    p_pv_hor_dict: Dict 
        Dictionary that consists of a the pv data for all the concerned 
        bus, defined over the MPC horizon
    p_load_hor_dict: Dict 
        Dictionary that consists of a the load demand data for all the  
        concerned bus, defined over the MPC horizon
    p_ev_chrg_f: Dict 
        Dictionary that consists of a the decision variable p_ev demand 
        for all the  concerned bus, defined over the MPC horizonn
        
        
    outputs: 
    [ ] : list
        Energy mismatch for each instant of the  Current imbalance period
    """
    
    
    e_mis_curT = [] # e_mis_curT : List to store Energy mismatch for each instant of the
                  #Current imbalance period
        
    # Import variables from modules probVar 
    Δt = probVar.Δt 
    ΔT = probVar.ΔT
    s_base = probVar.s_base
    
    # Indexes of the initial and last instant to consider in the current imbalance period 
    # varies in [0, mpcHrz[ i.e. defined in relation to the MPC horizon 
#     t_init = max(cur_k ,cur_T*ΔT - cur_k)
    t_init = max(0,cur_T*ΔT - cur_k)
    t_end = min(ΔT*(cur_T+1) - cur_k, mpcHrz)
    
    # TODO: remove the first for loop and indexed with t and index variable with 
    # [t_init: t_end] insteead
    for t in range(t_init,t_end):
        cur_T_network_cons = 0 #Total network consumption for the current imbalance period
                
        for cur_bus in range(0, totBus_network+1):# For all the bus in the network 
            if cur_bus in p_pv_hor_dict:
                cur_T_network_cons =  cur_T_network_cons + (Δt*p_pv_hor_dict[cur_bus][t]*s_base)/(1000*60)
                
            if cur_bus in p_load_hor_dict:
                cur_T_network_cons =  cur_T_network_cons - (Δt*p_load_hor_dict[cur_bus][t]*s_base)/(1000*60)
                
            if cur_bus in p_ev_chrg_f:
                cur_T_network_cons =  cur_T_network_cons - (Δt*p_ev_chrg_f[cur_bus][0, t]*s_base)/(1000*60)
             
        e_mis_curT.append(( (1/60)*Δt*p_plan_curT - cur_T_network_cons)) # Save the the e_mismatch 
                                                                         #for each instant of the imb per
        
    return e_mis_curT





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def prob_imbalancePer_Emis(cur_k, mpcHrz, totBus_network, p_plan, p_pv_hor_dict, 
                           p_load_hor_dict, p_ev_chrg, zero_to_k_Emis):
    
    """Computes the energy mismatch of each imbalance period defined over the MPC horizon
    
    Inputs:
    -------
    cur_k : Int 
         Current instant of the simulation
    mpcHrz : Int
        MPC horizon
    totBus_network : Int
        Total number of Bus in the network
    p_plan: array  
        Engagement, i.e. planned power for each imbalance period, defined over the problem horizon
    p_pv_hor_dict: Dict 
        Dictionary that consists of a the pv data for all the concerned 
        bus, defined over the MPC horizon
    p_load_hor_dict: Dict 
        Dictionary that consists of a the load demand data for all the  
        concerned bus, defined over the MPC horizon
    p_ev_chrg: Dict 
        Dictionary that consists of a the decision variable p_ev demand 
        for all the  concerned bus, defined over the MPC horizon
    zero_to_k_Emis: List
        List of already computed energy mismatch. Defined for each instant in [0, k[
        
        
    outputs: tuple 
   (prob_Emis, cur_T_Emis_init ): list
        prob_Emis: 
            Energy mismatch for each imbalance period over the MPC's horizon
        cur_T_Emis_init : 
            Energy mismatch of all the instants in the initial Imbalance period
    """
    
    
    prob_Emis = [] # list to store energy mismatch of each imbalance period
    
    T_init_hor = cur_k//probVar.ΔT         #index of Initial imbalance period over the MPC horizon 
    T_end_hor  = (cur_k+mpcHrz-1)//probVar.ΔT #index of final imbalance period over the MPC horizon 
    
    # for each imbalance period
    for T in range(T_init_hor,T_end_hor+1):
        
        # On the current imbalance period  T, t_in and t_en are defined in relation to the the problem horizon, 
        # unlike t_init and t_end in mpc.imbalancePer_Emis(args) defined in relation to the MPC horizon
        # Both t_in and t_en are needed to consider the E_mis of previous instants in the current imbalance period
        t_in = probVar.ΔT*T # t_in is the index of the first instant of the current imbalance period
        t_en = cur_k        
        
        # compute the imbalance Energy mismatch for each instant (starting from k ) of the current imbalance period
        cur_T_Emis = imbalancePer_Emis(T, cur_k, mpcHrz, totBus_network, p_plan[T], p_pv_hor_dict, p_load_hor_dict, p_ev_chrg)
        
        # sum energy mismatch of all the instant (starting from k) of the current imbalance period and add its 
        # absolute value to the energy mismatch list (starting from t_in up to k-1)
        prob_Emis.append(  cp.abs( sum(cur_T_Emis) + sum(zero_to_k_Emis[t_in:t_en]) )  ) 

        
        # Store energy mismatch of the initial imbalance period
        # It will be used later to retrieve the value 
        if T == T_init_hor:
            cur_T_Emis_init = cur_T_Emis
            
    return prob_Emis, cur_T_Emis_init
    

    

    
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    
    
def solve_prob (opt_cost, constr, solver_var):
    """ Solve optimization problem
    
    Inputs
    ------
    opt_cost: cvxpy variable
        Optimisation cost
    constr: cvxpy variable list 
        List of problem constraints
    solver_var: tuple, solver instance
       (0) mosek_solver, i.e cvxpy Mosek instance
       (1) gurobi_solver, i.e. cvxpy gurobi instance
       
    Output: 
    -------
    Optimisation problem solved 
    
    """
#     unpack input variables

    
    mosek_solver, gurobi_solver = solver_var
    
    prob = cp.Problem(cp.Minimize(opt_cost), constr) # define problem
    
    # If mosek fails to solve problem use Gurobi
    try: 
        prob.solve(solver = mosek_solver)
    except cp.error.SolverError :
        print('Using Gurobi')
        prob.solve(solver = gurobi_solver)
        
        
    return prob





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def init_EvArrival_soc (cur_k,  ev_data_curBus, soc_prob_curBus, mpc_sldFreq=1):
    """
    Initialise EV's (connected to the current bus) soc if its arrival time 
    is at the next loop iteration i.e. next instant cur_k+1
    
    Inputs
    -------
    cur_k : Int 
         Current instant of the simulation
    ev_data_curBus: Dict
        Dictionnoary containnig all the parameters of the Ev
        connected to the current bus
    soc_prob_curBus: 1D array, (Modified in place) 
        Soc of Ev connected to the current bus. 
        Defined over the problem horizon 
    mpc_sldFreq: Int, default=1
        MPC's sliding frequency
        
        
    Output
    ------
    socAtArr_updating_state: Bolean
        True:  Soc at arrival time has been updated 
        False: Soc at arrival time has not been updated 
    
    """
    socAtArr_updating_state = False # SOc at arrival updating state: Signals if the soc at arrival of Ev on cur_bus
                                    # is updated within function
    
    for curr_int in range(2, ev_data_curBus['nb_arr_dep']+1): # curr_int: Current Inteval
    # if next loop iteration (cur_k+1), equals  arrival time use soc of the concerned arrival time 
        arrival = ev_data_curBus['t_arr'+str(curr_int)]
        
        if (cur_k + 1 == arrival): 
            socAtArr_updating_state = True 
            soc_prob_curBus[0,cur_k+1: cur_k+1+mpc_sldFreq] = ev_data_curBus['soc_arr'+str(curr_int)]
            
            break# if code enter the if condition, no need to continue the for loop iteration 
                 # since the Ev can arrive only once on the next iteration i.e. curk+1 
                
    return socAtArr_updating_state





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________    

def init_Ev_soc (cur_k, soc_updating_state,  ev_plugStatus, soc_var, mpc_sldFreq=1):
    """
    Initialise EV's (connected to the current bus) soc for next loop iteration 
    i.e. next instant cur_k+1

    cur_k : Int 
         Current instant of the simulation
    soc_updating_state : Boolean 
         Signals whether the soc  at next loop iteration (cur_k+1), has (==>True) 
         or hasn't(==>False) been already updated. Output of function << mpc.init_EvArrival_soc(args)>>
    ev_plugStatus: Int 
        First output of the function <<brp.interval_finder(cur_k,*)>>. Used to check whether 
        the ev on current bus is plugged-in at the current instant cur_k
    soc_var: tuple
        (0) soc_prob_curBus: 1D array, (Modified in place) 
                Soc of the Ev connected to the current bus. 
                Defined over the problem horizon
        (1) soc_curBus: 1D cvxpy array
                Predicted Soc of the EV connected to the current bus.
                Defined over the MPC horizon
    mpc_sldFreq: Int, default=1
        MPC's sliding frequency

    """
    # unpacking inputs 
    soc_prob_curBus, soc_curBus = soc_var
    
    # If SoC not updated yet for the next loop excecution do it using the opt result 
    # or soc_min when not connected  to the grid
    if soc_updating_state == False :

        if ev_plugStatus == 0: # If ev is not connected to the grid 
            soc_prob_curBus[0,cur_k+1: cur_k+1+mpc_sldFreq] = probVar.soc_min
        else: 
            soc_prob_curBus[0,cur_k+1:cur_k+1+mpc_sldFreq] = soc_curBus.value[0,1:1+1]




            
#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________           
          
def pred_Irradiance(cur_k, mpcHrz_in_min, h_back_windows, h_frcst_freq, cur_date, file_loc, day_type, scale=150 ):
    """
    Define P_Pv over the MPC horizon based on the daytype (day regime) that produces the 
    lowest Mean Squared Error (MSE). 
    
    
    Inputs:
    -------
    cur_k: Int
        Current simulation step k 
    mpcHrz_in_min : Int
        MPC horizon in minutes
    h_back_windows: Float 
        Size of the the backward facing windows (that is used to compute the MSE )
        Egg: 1 ==> 1hour, 0.25 ==> 15mn
        NB: make sure its value converted in minute is a multiple of 5 otherwise produce error
    h_frcst_freq: Float
        Updating frequency of the prediction i.e. at which interval to recompute the MSE
        Egg: 1 ==> 1hour, 0.25 ==> 15mn
        NB: Must be equal or higher than the MPC problem temporal resolution Δt
    curr_date: String in format '2020 month day'
       Current day of simulation '2020 03 5'
    file_loc: File 
        Define the location of the file containing the regime switching 
        P_pv parameters
    day_type: Int
        Type of regime to suppose before the first update occurs
            0 ==> Clear day
            1 ==> Overcast day
            2 ==> Mild day 
            3 ==> Moderate day
    scale: Float, default: 150
        Value use to upscale (to convert) irradiance in power 
        


    """
    
     # Load file containig parameters 
    file_to_read = open(file_loc, "rb")
    extracted_data = pickle.load(file_to_read)
    
    
    data2020 = extracted_data['2020_data'] # Extract 2020 data from file
    per_start_daylight, per_end_daylight = extracted_data['per_daylight'] # Unpack daylight periods


    nb_sim_days = 1 # Define numbers of simulation days
    freqq = 5 # mn Dataset  is sampled at 5 mn
    day_totPeriod = probVar.fullDay_min //freqq 
    
    # definition of the size of the backward facing windows in period 
    per_back_windows = int(h_back_windows*60//freqq)  # Convert hour in period
    
    # Definition of the updating frequency
    per_frcst_freq = int(h_frcst_freq*60//freqq)      # Convert hour in period

    # Define period where forcasting occurs at each day
    frcst_per_list = list(map(int, np.arange(per_start_daylight, per_end_daylight, per_frcst_freq)[1:])) 

    # Define the size of the predicted GHI over the number of day of simulation (+1 due to 24H MPC horizon)
    pred_GHI = np.zeros(((nb_sim_days+1)*day_totPeriod))
    
    
    # define a date range
    DAYS = pd.date_range(start=cur_date, periods=nb_sim_days, freq='D')
    DAYS = DAYS + DAYS.freq                    # Simulation always starts at the second day
    # Icrease day count depending on the cur_k 
    # Egg: for probVar.Δt = 15mn, lenght of a day = 96
    # if cur_k = 100 ==> 4th period of the second day of simulation so day count is increased by one day
    day_count = cur_k//(probVar.fullDay_min//probVar.Δt)
    DAYS = DAYS + day_count*DAYS.freq
    
    
    for day_index, cur_day in enumerate(DAYS):# only one day 
        
        #Define days 
        day_prev = cur_day - cur_day.freq   # previous day
        day_init = cur_day                  # current day
        day_future = cur_day + cur_day.freq # future day

        #Extract previous and Current day real data 
        prevDay_data =  data2020[(data2020.index >= day_prev) & (data2020.index < day_init)]
        curDay_data = data2020[(data2020.index >= day_init) & (data2020.index < day_future)]

        # Define predicted values dependant on the type of regime for the current day 
        curDay_data_dict = rssp.dayType_dataPred(extracted_data['model_params'], extracted_data['per_daylight'], curDay_data )

        day_type = 0 # Initialize current day regime
        pred_GHI = rssp.initialize_predGHI(extracted_data['model_params'], extracted_data['per_daylight'], 
                                           prevDay_data, pred_GHI, day_type)

        d0_index = day_index               # lower bound current day
        
        cur_k = cur_k%probVar.per_day      # Convert cur_k so that it is always defined within the interval [0, probVar.per_day]
        int_var = probVar.Δt//freqq        # An itermedaite variable, gives how many periods of freqq is in probVar.Δt
        d1_index = (cur_k+1)*int_var       # Convert the cur_k from index [0, 96] (if probVar.Δt = 15mn) OR [0, 144] (if probVar.Δt = 10mn)
                                           # to indice following a freqq = 5mn 

            
        for k in range (d0_index, d1_index): # for all the instant of the current day up to the current instant 
            pred_GHI, day_type  = rssp.pred_Irradiance_updater(k, d0_index, frcst_per_list, per_back_windows, pred_GHI,
                                                               extracted_data['per_daylight'], 
                                                               (curDay_data, curDay_data_dict), 
                                                               day_type ) 
            
        # Set bound of the MPC horizon in periods 
        index0 = d1_index-int_var              
        index1 = index0 + mpcHrz_in_min//freqq+int_var
        
        # Extract predicted irradiance for only periode covered by the MPC horizon
        pred_output = pred_GHI[index0: index1]
        
        # create a datarange using the dataset frequency that is 5mn 
        cur_day_data_range = pd.period_range(start=cur_day, periods = len(pred_GHI), freq='5T')
        
        # put the solar prediction into a dataframe
        pred_output = pd.DataFrame(data=pred_output, index= cur_day_data_range[index0: index1]) 
        
        # Resample the previous data frame using the study temporal resolution probVar.ds_freq_str
        # and return that said value
        
    return (pred_output.resample(probVar.ds_freq_str).mean()/scale).T





#________________________________________________________________________________________________________________________  
#------------------------------------------------------------------------------------------------------------------------ 
#________________________________________________________________________________________________________________________           

def soc_satisfaction (ev_data_dict, soc_prob_dict, soc_prob_dict_dumb):
    """ 
    Compute the mean satisfaction of attaining the minimum soc at departure for all the EVs in the network. 
    In the best case scenario, at all the departure instants the reached soc must be >= soc_dumb ==> and 
    the function output (i.E. mean over all the departure instants considered) for the associated ev equals 0. 
    In the worst case, at all the departure instants the reached soc = soc_arrival_dumb ==> and 
    the function output (i.E. mean over all the departure instants considered) for the associated ev on ]0, 1[
    
    
    Inputs: 
    -------
    ev_data_dict: Dict
        Dictionnoary containnig all the parameters of the Evs connected 
        to the network;  Output of function << brp.init_evDataParams(args) >>
    soc_prob_dict: Dict 1D float array
                Optimal value of Decision variable soc, defined over the problem horizon
    soc_prob_dict_dumb: Dict 1D float array
                Value of Decision variable soc, defined over the problem horizon, when using the dumb strategy



    Output: Tuple of list
    -------
        (0) sas_rate_list : list of mean (over all the departure instants considered) of satisfaction rate for each Ev
        (1) worst sas_rate_list : list of mean (over all the departure instants considered) of worst satisfaction rate 
        for each Ev i.e., if the considered ev didn't charge at all when it was connected. 
        Nb: If there is no departure instants where soc_dumb_dep > soc_mpc_dep (Best scenario case), the first output 
        sas_rate_list is empty, its mean is not defined so the function will output 0. Same goes for the second output
        sas_rate_list as well.
    """


    dep = 't_dep'
    soc_arr = 'soc_arr'
    # List to store final results 
    diff_mean_norm_list, diff_mean_not_norm_list, mean_norm_term_list = [], [], []


    for cur_bus in ev_data_dict.keys(): 
        # defining some list 
        elm_list, index_list = [], []

        # Get soc at departure time for MPC
        mpc_soc = [soc_prob_dict[cur_bus][0,int(ev_data_dict[cur_bus][dep+str(cur_dep)])]
                   for cur_dep in range (1,ev_data_dict[cur_bus]['nb_arr_dep']) ]

        # Get soc at departure time for Dumb Strategy
        dumb_soc = [soc_prob_dict_dumb[cur_bus][int(ev_data_dict[cur_bus][dep+str(cur)])] 
                    for cur in range (1,ev_data_dict[cur_bus]['nb_arr_dep']) ]

        # Compute diff between both i.e. pos part of (dumb_soc - mpc_soc )
        res1 = np.maximum(0, np.array(dumb_soc) - np.array(mpc_soc))
#         print(cur_bus, "   ", res1)
        # Get index (+1 because departure indexation starts at 1 and not at 0) of departure time where 
        # the diffenrence is significant (1e-3, signicance level) 
        [(index_list.append(ind+1), elm_list.append(elm)) for ind, elm in enumerate(res1) if elm >=1e-2] # 1e-2 % of soc

        # get the departure soc of the instant to consider
        dumb_soc2_consider = np.array(dumb_soc)[list(np.array(index_list)-1)]
#         print(cur_bus, "   ", len(dumb_soc2_consider))

        # Compute normalization term given by the diffenrence between the soc_at_departure and the 
        # soc_at_arrival using the dumb strategy, (worst satisfaction rate  scenario if the ev doesn't charge at
        # all during the time it is connected.
        norm_term = [dep_soc - ev_data_dict[cur_bus][soc_arr+str(elm)] 
                     for elm, dep_soc  in zip(index_list, dumb_soc2_consider)]

        # Compute the normalized soc difference for each departure instant considered
        norm_diff = [diff/norm_elm for diff, norm_elm in zip(elm_list, norm_term)]

        # Compute the mean difference with normalization
        diff_mean_norm =  np.nan_to_num(np.mean(norm_diff)) # If empty list, Replace nan given mean by 0

        # Compute the mean difference without normalization
        diff_mean_not_norm = np.nan_to_num(np.mean(elm_list))  # If empty list, Replace nan given mean by 0

        # store results for each Ev connected to a bus
        mean_norm_term_list.append(np.nan_to_num(np.mean(norm_term))) # The mean of normalized term i.e. upper bound
                                                                      # of  diff_mean_not_norm  worst case 
        diff_mean_norm_list.append(diff_mean_norm)
        diff_mean_not_norm_list.append(diff_mean_not_norm)
        
    return diff_mean_not_norm_list, mean_norm_term_list