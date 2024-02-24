# This project implements an LSTM that predicts rainfall for a variety of countries

General_LSTM.py - Original code, trained on each file separately

General_LSTM_op.py - Optimized code, trained on all files, 365 rows (a page) at a time

TODO: Each time we select a random month (30 pages) from all regions ---- DONE

TODO: Modify code to adapt `create_dataset` function; cap the maximum pages in training (prob 10k, `batchsize=200`, 50 batches); store the trained pages global data structure somewhere

