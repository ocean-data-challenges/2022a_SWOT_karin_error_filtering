# Description of CNN experiment (eNATL60-BLB002)

<div style="text-align: right"><i> 2022-09-20 CNN SETUP </i></div>

***
**Authors:**  Anaëlle Tréboutte, Benjamin Carpentier
Pierre Prandi, Yannice Faugere (CLS), Gerald Dibarboure (CNES) <br>
**Copyright:** 2022 CLS & Datlas <br>
**License:** MIT



### Simulated SWOT Data:
- Swath generated with [`SWOTSimulator`](https://github.com/CNES/swot_simulator) software
- Input model: [`eNATL60-BLB002`](https://github.com/ocean-next/eNATL60) numerical simulation WITHOUT explicit tidal forcing;  
- Noisy SSH = true SSH + Karin noise modulated by the waves

### Waves model : 
- Global ocean reanalysis wave system of Météo-France ([`WAVERYS`](https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-032.pdf)) with a resolution of 1/5° degree 

### Division of the dataset : 
- Year 2009 : training dataset (train : **75 %**, validating : **25 %**)
- Year 2010 : dataset for the calculation of scores (Note: in the data challenge we consider only cycle 13 in 2010 for the validation)

### Data preprocessing :  
- used of anomalies of SSH 
- used of data normalization
- used of data augmentation : Vertical and/or Horizontal Flip

### Division of the swaths : 
- 512 km along-track


### Filtered data folder:
- eNATL60-BLB002/cnn