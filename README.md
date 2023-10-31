# EqShake

The accurate estimation of earthquake magnitude is crucial for assessing seismic hazards and ensuring effective disaster mitigation strategies. EqShake is a deep-learning model for accurate earthquake magnitude estimation utilizing single-station raw waveforms. EqShake, as opposed to other available models, is designed to be independent of event-station distance.

Model Structure
--
**EqShake** takes advantage of both P and S waves for magnitude estimation. As only the first 3 seconds of the body waves are used as inputs, **EqShake** works for waveforms recorded at any distance in local scales. Furthermore, event-station distance is fed to the model to account for the attenuation.

<div id="header" align="center">
  <img src='mag.jpg' width='500'>
</div>

Model Performance
--
The test data included here consists of 30k samples. **EqShake** demonstrates excellent performance with R2-score= 0.93, MSE= 0.037, Mean= 0.0, and std= 0.22. 

<div id="header" align="center">
  <img src='predictions.jpg' width='500'>
</div>

<div id="header" align="center">
  <img src='mag_hist.jpg' width='500'>
</div>
