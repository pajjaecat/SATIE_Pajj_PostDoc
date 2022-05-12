## Description of all the notebooks contained in this folder


All the files called within the notebooks are located in [DataFiles](../DataFiles). When executing, make sure the file location is rightly set.


</br> 

***


#### [SolarPre_DataExtraction](SolarPre_DataExtraction.ipynb)
Solar irradiance prediction based on different types of day/regimes (clear, Overcast, Mild, Moderate, High ) using [fourrier expansions](https://en.wikipedia.org/wiki/Fourier_series). Following the work of [SAyMSe2016](https://ieeexplore.ieee.org/document/7855546), the solar irradiance is predicted at a fixed hour daily using constant backwards-looking windows. We improve upon the previous method by using variable-length backwards-looking windows and updating the prediction at regular intervals throughout the day. 


