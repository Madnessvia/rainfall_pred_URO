# Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import scipy.stats
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
import json

# Parameters
TargetLabel = 'streamflow_mmd'
LearningRate = 0.001
TIME_STEP = 365
EPOCHs = 75
BatchSize = 200
Patience = 50
TrainRatio = 0.4
ValidationRatio = 0.2

# Input columns
f_columns =['mean_temperature_C', 'precipitation_mmd', 'pet_mmd']
staticColumns=['area_km2','mean_elevation_m','mean_slope_mkm','shallow_soil_hydc_md','soil_hydc_md','soil_porosity','depth_to_bedrock_m','maximum_water_content_m','bedrock_hydc_md','soil_bedrock_hydc_ratio','mean_precipitation_mmd','mean_pet_mmd','aridity','snow_fraction','seasonality','high_P_freq_daysyear','low_P_freq_daysyear','high_P_dur_day','low_P_dur_day','mean_forest_fraction_percent']

# Input folder (daily csv files)
folder = 'CAMELS-US/Daily_Data/'

# Output folder, where we save the results
outputfolder = 'CAMELS-US/Output_USCA/'

if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)
    print('Oops! directory did not exist, but no worries, I created it!')

SaveModel = outputfolder

#Static Data- it must contain items listed by "staticColumns" and grid code
path_static = 'CAMELS-US/Attributes/attributes.csv'

# Read and Normalize statistical features
dfs = pd.read_csv(path_static)  # Static Data
OurDesiredStaticAttributes = dfs.columns
f_transformer = StandardScaler()
f_transformer = f_transformer.fit(dfs[OurDesiredStaticAttributes].to_numpy())
dfs.loc[:, OurDesiredStaticAttributes] = f_transformer.transform(
  dfs[OurDesiredStaticAttributes].to_numpy()
)
dftemp = pd.read_csv(path_static)
dfs['gridcode'] = dftemp['gridcode'].astype(str)

# Create Dataset
def create_dataset(X, y, date_df, doy_df, time_steps=1):
    Xs, ys, date, doy = [], [], [], []
    for i in range(len(X) - time_steps):
        X_seq = X.iloc[i:(i + time_steps)]

        # Check if there's any NaN in the X sequence or the corresponding y value
        if not X_seq.isnull().values.any() and not pd.isnull(y.iloc[i + time_steps-1]):
            Xs.append(X_seq.values)
            ys.append(y.iloc[i + time_steps-1])
            date.append(date_df.iloc[i + time_steps-1])
            doy.append(doy_df.iloc[i + time_steps-1])

    return np.array(Xs), np.array(ys), np.array(date), np.array(doy)


# NSE function (Nash-Sutcliff-Efficiency)
def NSE(targets,predictions):
  return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))

# model definition
model = keras.Sequential()
model.add(keras.layers.LSTM(units=256, return_sequences=False, input_shape=(TIME_STEP, len(f_columns) + len(staticColumns))))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))
callbacks = [keras.callbacks.EarlyStopping(patience=Patience, restore_best_weights=True)]
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LearningRate))     # compile a model based on MSE

# count number of files
def count_files(directory):
    entries = os.listdir(directory)
    file_count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)
    return file_count

num_pages = 30
train_size = int(count_files(folder) * TrainRatio) * num_pages
val_size = int(count_files(folder) * ValidationRatio) * num_pages
# train_size = 10000
# val_size = 5000
test_size = train_size

# number of rows in each file
max_days = 14610
useful_days = max_days - TIME_STEP
iterations = int(useful_days / num_pages)

# Global data structure to store trained data
file_path = 'CAMELS-US/Output_USCA/used_days.json'
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        used_days = json.load(file)
    print("JSON file already exists and read into RAM")
else:
    used_days = {}
    print("JSON file created for the first time")


# total_used_days = sum(len(lst) for lst in used_days.values())
# iteration = int(total_used_days / (len(used_days.keys()) * 30))

h5_files = [f for f in os.listdir(SaveModel) if f.endswith('.h5')]
if h5_files:
    h5_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    latest_h5_file = os.path.join(SaveModel, h5_files[0])
    iteration = int(h5_files[0].split('_')[1]) + 1
    model.load_weights(latest_h5_file)
    print(f"Loaded weights from {latest_h5_file}")
else:
    iteration = 1
    print("No .h5 files found, starting training from scratch.")

all_files = {}
for file in os.listdir(folder):
    filename = file.rstrip(".csv")
    Dir = folder + str(file)
    dataframe = pd.read_csv(Dir)
    all_files[filename] = dataframe

while (iteration <= iterations):
    iter = 0
    X_train, y_train = [], []
    X_val, y_val = [], []
    # X_test, y_test = [], []
    print(f"Iteration {iteration}/{iterations}")
    
    for GridCode in all_files.keys():
        if GridCode not in used_days.keys():
            used_days[GridCode] = []

        df = all_files[GridCode].copy()
        df['date'] = pd.to_datetime(df.pop('date'))
        df['day_of_year'] = df['date'].dt.dayofyear
        df[TargetLabel] = np.log1p(df[TargetLabel])
        
        possible_starts = [i for i in range(useful_days) if i not in used_days[GridCode]]

        start_days = random.sample(possible_starts, num_pages)
        used_days[GridCode].extend(start_days)

        for start_day in start_days:
            data = df.iloc[start_day:start_day+TIME_STEP].copy()
            f_transformer = StandardScaler().fit(data[f_columns])
            data[f_columns] = f_transformer.transform(data[f_columns])
        
            static_row = dfs[dfs['gridcode'] == GridCode]
            for item in staticColumns:
                data.loc[:, item] = static_row[item].to_numpy()[0]
            
            input_columns = f_columns + staticColumns
        
            if iter < train_size:
                if not data.isnull().values.any() and not pd.isnull(data[TargetLabel].iloc[TIME_STEP-1]):
                    X_train.append(data[input_columns].values)
                    y_train.append(data[TargetLabel].iloc[TIME_STEP-1])
            elif iter < train_size + val_size:
                if not data.isnull().values.any() and not pd.isnull(data[TargetLabel].iloc[TIME_STEP-1]):
                    X_val.append(data[input_columns].values)
                    y_val.append(data[TargetLabel].iloc[TIME_STEP-1])
            else:
                break
                # X_test.append(data[input_columns].values)
                # y_test.append(data[TargetLabel].iloc[TIME_STEP-1])
            iter += 1
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    # X_test, y_test = np.array(X_test), np.array(y_test)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHs,
        batch_size=BatchSize,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=callbacks
        )
    
    with open(file_path, 'w') as f:
        json.dump(used_days, f)

    # np.savetxt(SaveModel + 'Pages_Based_On_Which_Trained_sofar.out', used_days, delimiter=',')
    path = SaveModel + 'interationNum_' + str(int(iteration))+'_Generally_Trained_UP_TO_NOW_Model' + '.h5'
    model.save_weights(path)

    iteration += 1
    
path = SaveModel + 'Generally_Trained_Model' +'.h5'
model.save_weights(path)