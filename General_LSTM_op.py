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
# folder = 'Data_Daily_Clustered_based_on_AI_SF_SI/Cluster_' + str(int(cluster)) + '/'
folder = '/home/xlhdesktop/rainfall_pred_URO/CAMELS-US/Daily_Data/'

# Output folder, where we save the results
# ourputfolder = 'Output_USCA/outputs_with_seasonality_removed_and_p_and_pet_as_inputs/general_model_cluster_'       + str(int(cluster)) +
outputfolder = '/home/xlhdesktop/rainfall_pred_URO/CAMELS-US/Output_USCA/'

if not os.path.exists(outputfolder):

    os.makedirs(outputfolder)
    print('Oops! directory did not exist, but no worries, I created it!')


SaveModel = outputfolder

#Static Data- it must contain items listed by "staticColumns" and grid code
path_static = '/home/xlhdesktop/rainfall_pred_URO/CAMELS-US/Attributes/attributes.csv'

# Read and Normalize statistical features
dfs = pd.read_csv(path_static)  # Static Data
OurDesiredStaticAttributes = dfs.columns
f_transformer = StandardScaler()
f_transformer = f_transformer.fit(dfs[OurDesiredStaticAttributes].to_numpy())
dfs.loc[:, OurDesiredStaticAttributes] = f_transformer.transform(
  dfs[OurDesiredStaticAttributes].to_numpy()
)
dftemp = pd.read_csv(path_static)
dfs['gridcode'] = dftemp['gridcode']

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
# model.add(keras.layers.LSTM(units=256, return_sequences=False, input_shape=(TIME_STEP, X_train.shape[2])))
model.add(keras.layers.LSTM(units=256, return_sequences=False, input_shape=(TIME_STEP, 23)))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))

callbacks = [keras.callbacks.EarlyStopping(patience=Patience, restore_best_weights=True)]

model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LearningRate))     # compile a model based on MSE

# all_X_train, all_y_train = [], []
# all_X_train_val, all_y_train_val = [], []
# all_X_test, all_y_test = [], []
# all_X_val, all_y_val = [], []

def count_files(directory):
    entries = os.listdir(directory)
    file_count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)
    return file_count

train_size = int(count_files(folder) * TrainRatio) * 30
val_size = int(count_files(folder) * ValidationRatio) * 30
test_size = train_size

# TraindGridCodes = np.array([])
TrainedPages = np.array([])

max_days = 14610
useful_days = max_days - TIME_STEP
iterations = int(useful_days / 30)
used_days = {}

for iteration in range(iterations):
    iter = 0
    X_train, y_train = [], []
    X_val, y_val = [], []
    # X_test, y_test = [], []
    print(f"Iteration {iteration+1}/{iterations}")

    for file in os.listdir(folder):
        GridCode = int(file.rstrip(".csv"))
        if GridCode not in used_days:
            used_days[GridCode] = []

        Dir = folder + str(file)
        
        # df = pd.read_csv(Dir).dropna()
        df = pd.read_csv(Dir).fillna(0)
        df['date'] = pd.to_datetime(df.pop('date'))
        df['day_of_year'] = df['date'].dt.dayofyear
        df[TargetLabel] = np.log1p(df[TargetLabel])
        
        possible_starts = [i for i in range(useful_days) if i not in used_days[GridCode]]

        start_days = random.sample(possible_starts, 30)
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
                X_train.append(data[input_columns].values)
                y_train.append(data[TargetLabel].iloc[TIME_STEP-1])
            elif iter < train_size + val_size:
                X_val.append(data[input_columns].values)
                y_val.append(data[TargetLabel].iloc[TIME_STEP-1])
            # else:
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
    
    # np.savetxt(SaveModel + 'Pages_Based_On_Which_Trained_sofar.out', used_days, delimiter=',')
    path = SaveModel + 'interationNum_' + str(int(iteration))+'_Generally_Trained_UP_TO_NOW_Model' + '.h5'
    model.save_weights(path)
    
path = SaveModel + 'Generally_Trained_Model' +'.h5'
model.save_weights(path)