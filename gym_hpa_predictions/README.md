# Predictions in gym-hpa

This is an extension of the original gym-hpa framework, were we attempt to add predictions of certain metrics in the agent's observation space.  


## What kind of algorithms are used?

<b>Naive:</b> The naive method is a simple method, used commonly as a benchmark. The prediction is the same as the last observed value. Despite its simplicity, in environments with too much randomness, it performs very well, usually outperforming more sophisticated and complex methods.
<br><br><b>Simple Exponential Smoothening (SES):</b>SES is primarily used in timeseries that lack any trend. It smoothens the values based on previous errors and observations, giving exponentially reduced weights in past observations.
It has only one parameter, a, getting values from 0 to 1. Higher the value, higher the effect of the previous errors in the prediction, which adds an instability to the predictions. High values are preferred in environments with high randomness.
<br><br><b>LSTM</b>: LSTMs are neural networks, used extensively in forecasting due to fact that they are able to take into account past observations and predictions they made. 
<br><br><b>ARIMA</b>: ARIMA models make prediction on stationary timeseries using weights on previous values. 
It has three parameters, p, d and q, which can be determined by the auto_arima() function automatically, though it is computationally intensive. If the timeseries is not stationary, it also converts it to a stationary before it makes the predictions. It is usually preferred for short term forecasting, not long term, due to the fact that for longer forecasting horizons possible errors get accumulated.

## How were the predictions implemented in the main project?
All the prediction algorithms were implemented in a single class, so as to be more easily deployed and used in the main project, with minimum changes in its original code. There are two ways of forecasting implemented:
<br><b>In-sample Predictions:</b> They are predictions done using the given dataset. This way, they give an estimation of the model’s accuracy during the simulation or cluster. Also, because these environments might be more chaotic than the dataset’s, it is easier to predict.
<br><br><b>Out-of-sample Predictions</b>: This is the expansion of the first idea, where predictions take place in real time, during the simulation/cluster mode. It is what was originally intended. 
<br>There is also a forecasting horizon, so that the algorithms can be used for predictions further into the future.
## Team

* [Jose Santos](https://scholar.google.com/citations?hl=en&user=57EIYWcAAAAJ)

* [Tim Wauters](https://scholar.google.com/citations?hl=en&user=Kvxp9iYAAAAJ)

* [Bruno Volckaert](https://scholar.google.com/citations?hl=en&user=NIILGOMAAAAJ)

* [Filip de Turck](https://scholar.google.com/citations?hl=en&user=-HXXnmEAAAAJ)

## License

Copyright (c) 2020 Ghent University and IMEC vzw.

Address: IDLab, Ghent University, iGent Toren, Technologiepark-Zwijnaarde 126 B-9052 Gent, Belgium 

Email: info@imec.be.


