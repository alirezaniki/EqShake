# EqShake

The accurate estimation of earthquake magnitude is crucial for assessing seismic hazards and ensuring effective disaster mitigation strategies. **EqShake** is a deep-learning model for accurate earthquake magnitude estimation using single-station raw waveforms. As opposed to other available models, **EqShake** is designed to be independent of waveform length, applicable to events recorded at local scales (<3 deg). **EqShake** was trained on 140k samples from STEAD dataset (https://github.com/smousavi05/STEAD).

Model Structure
--
**EqShake** takes advantage of both P and S waves for magnitude estimation. As only the first 3 seconds of the body waves are used as the input, **EqShake** works for waveforms with P-S separation larger than 3 seconds. Furthermore, event-station distance is fed to the model to account for the attenuation.

<div id="header" align="center">
  <img src='mag.jpg' width='500'>
</div>

Model Performance
--
The test dataset included here consists of 30k samples. **EqShake** demonstrates excellent performance with R2-score= 0.93, MSE= 0.037, Mean= 0.0, and std= 0.22. 

<div id="header" align="center">
  <img src='predictions.jpg' width='500'>
</div>

<div id="header" align="center">
  <img src='mag_hist.jpg' width='500'>
</div>
