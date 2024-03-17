# This project implements an LSTM that predicts rainfall for a variety of countries

General_LSTM.py - Original code, trained on each file separately

General_LSTM_monthly.py - Current version, random selecting 2 months from training region, 1 month from validation region, for every file.

The training can be terminated and restarted. It reads the trained months and select from months that are not yet trained. It also loads the weights from the last training iteration into the model if restarted.

TODO: 
1. Fit the model to 7000 csv files
2. Attention layer integration
3. Testing?
