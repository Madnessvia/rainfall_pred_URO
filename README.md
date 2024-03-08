# This project implements an LSTM that predicts rainfall for a variety of countries

General_LSTM.py - Original code, trained on each file separately

General_LSTM_op.py - Optimized code, trained on all files, 30*365 rows (30 pages) at a time

The training can be terminated and restarted. It reads the trained pages and select from pages that are not yet trained. It also loads the weights from the last training iteration into the model if restarted.

TODO: 
1. Read all files into RAM before training
2. Testing the model
