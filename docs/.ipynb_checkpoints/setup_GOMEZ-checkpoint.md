# Description of GOMEZ experiment (eNATL60-BLB002)

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

### Gomez et al. (2020) filter
Variational filter: Minimization of a cost function: 

$$𝐽(ℎ) = \frac{1}{2} ‖𝑚 \circ (ℎ -ℎ_{𝑜𝑏𝑠} )^2 ‖^2 + \frac{𝜆_2}{2} ‖∆ℎ‖^2$$

Goal: Optimize $𝜆_2$ via the use of RMSE and MSR (Mean Spectral Ratio calculated from PSD). 

Results : Better results than classical filters (boxcar, Gaussian, lanczos, loess) 

Disadvantages: Parameterization of $𝜆_2$ depending on: 
- The study area
- The season
- The data 

→ In this data-challenge, use of $𝝀_2$=𝟏𝟎 (Value adapted for areas of high variability)


### Filtered data folder:
- eNATL60-BLB002/gomez

### References
- Gomez-Navarro, L., Cosme, E., Sommer, J., Papadakis, N., Pascual, A., 2020. Development of an Image De-Noising Method in Preparation for the Surface Water and Ocean Topography Satellite Mission. Remote Sens. 12, 734. https://doi.org/10.3390/rs12040734
- Gómez Navarro, L., 2020. Image de-noising techniques to improve the observability of oceanic fine-scale dynamics by the SWOT mission. (Theses). Universite Grenoble Alpes; Universitat de les Illes Balears.
