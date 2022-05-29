All the files called within the notebooks here described, are located in [DataFiles](../DataFiles). When executing the notebooks, make sure the file location is rightly set (keep the same directory structure as this repository and it should work just fine) and that all the compressed files (.zip, .7z) are decompress in the corresponding folder. 
> As exemple decompress the content of the file [DataFiles/ElectricNation.zip](../DataFiles/ElectricNation.zip)  in a the same directory as  [DataFiles/ElectricNation/]


</br> 


## Brief description of each  notebook


***


#### [ElectricNation](ElectricNation.ipynb) 
> Extract Evs data distribution using the freely available ElectricNation [Dataset](https://www.westernpower.co.uk/electric-nation-data). A copy containing the main files that we make use of are available in the subfolder [DataFiles/ElectricNation.zip](../DataFiles/ElectricNation.zip).



#### [Entsoe](Entsoe.ipynb)
> Statistics using box plot for  imbalance price extracted from [here](https://transparency.entsoe.eu/balancing/r2/imbalance/show?name=&defaultValue=true&viewType=TABLE&areaType=COMBINED_IBA_IPA_SCA&atch=false&dateTime.dateTime=13.12.2021+00:00%7CCET%7CDAYTIMERANGE&dateTime.endDateTime=13.12.2021+00:00%7CCET%7CDAYTIMERANGE&marketArea.values=CTY%7C10YFR-RTE------C!SCA%7C10YFR-RTE------C&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)) for year 2019 and 2020.


#### [MSAR_SolarMeanPred](MSAR_SolarMeanPred.ipynb)
> Markov Switching Auto-Regressive (MSAR) Model predicts, based on past information, the mean value of the solar power over a defined number of minutes into the future. See [Basic Time_series_modelling](../PdfFiles/Basic_Time_series_modelling.pdf) for a brief introduction to the MSAR model.




#### [TestAnEv](TestAnEv.ipynb) 
> Extract Evs data distribution using the freely available Test-An-Ev [Dataset](http://mclabprojects.di.uniroma1.it/smarthgnew/Test-an-EV/?EV-code=EV1). A [script](../Modules/TestAnEvDataSet_DownloadingScript.py) (written by [@sharyal](https://github.com/sharyalZ)) is provided to download the said dataset and a copy is also available in the subfolder [DataFiles/TestAnEvDataset.zip](../DataFiles/TestAnEvDataset.zip) 



#### [SolarPre_DataExtraction](SolarPre_DataExtraction.ipynb)
> Solar irradiance prediction based on different types of day/regimes (clear, Overcast, Mild, Moderate, High ) using [fourrier expansions](https://en.wikipedia.org/wiki/Fourier_series). Following the work of [SAyMSe2016](https://ieeexplore.ieee.org/document/7855546), the solar irradiance is predicted at a fixed hour daily using constant backwards-looking windows. We improve upon the previous method by using variable-length backwards-looking windows and updating the prediction at regular intervals throughout the day. 

#### [V1G_MPC](V1G_MPC.ipynb)
> MPC V1G simulation based on the formulation proposed in [ProblemFormulation](../PdfFiles/MPC_Prob_Formulation.pdf).



#### [V1G_MPC_SimResults](V1G_MPC_SimResults.ipynb)
> The MPC V1G Problem described in [V1G_MPC](V1G_MPC.ipynb) has been solved for different parameters and inputs. The results can be easily accessed for a quick analysis using this notebook.

</br> 



