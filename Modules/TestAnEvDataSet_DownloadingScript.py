import pandas as pd
import sys,os


path = os.getcwd()
print ("The current working directory is", path)

# Directory 
directory = "Test"

# Parent Directory path
parent_dir = os.getcwd()
  
# Folder where to save data
path_folder = os.path.join(parent_dir, directory)
os.mkdir(path_folder)

folder_soc = "SoC"
folder_time = "Time"

# Create Subfolder named Soc and Time in path_folder/
path_soc = os.path.join(path_folder,folder_soc)
os.mkdir(path_soc)

path_times = os.path.join(path_folder,folder_time)
os.mkdir(path_times)


nb_evs = 185 # Total number of EV in the dataset
for i in range(1,nb_evs):
#     data_graph1 = pd.read_csv('http://smarthg.di.uniroma1.it/Test-an-EV/csv/EV' + str(i) +'.1.csv')
    data_graph2 = pd.read_csv('http://smarthg.di.uniroma1.it/Test-an-EV/csv/EV' + str(i) +'.2.csv')
    data_graph3 = pd.read_csv('http://smarthg.di.uniroma1.it/Test-an-EV/csv/EV' + str(i) +'.3.csv')
  

    ev_file_path_2 = os.path.join(path_soc, 'EV_' + str(i) + '_SoC.csv')
    ev_file_path_3 = os.path.join(path_times, 'EV_' + str(i) + '_Times.csv')
    
    data_graph2.to_csv(ev_file_path_2,index=False)
    data_graph3.to_csv(ev_file_path_3,index=False)
    
    print('EV number', i)