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

# Parameters
TargetLabel = 'streamflow_mmd'
LearningRate = 0.001
TIME_STEP = 365
EPOCHs =75
BatchSize = 200
Patience = 50
TrainRatio = 0.4
ValidationRatio = 0.2

# Input columns
f_columns =['mean_temperature_C', 'precipitation_mmd', 'pet_mmd']
staticColumns=['area_km2','mean_elevation_m','mean_slope_mkm','shallow_soil_hydc_md','soil_hydc_md','soil_porosity','depth_to_bedrock_m','maximum_water_content_m','bedrock_hydc_md','soil_bedrock_hydc_ratio','mean_precipitation_mmd','mean_pet_mmd','aridity','snow_fraction','seasonality','high_P_freq_daysyear','low_P_freq_daysyear','high_P_dur_day','low_P_dur_day','mean_forest_fraction_percent']
print(len(staticColumns))
# Input folder (daily csv files)
# folder = 'Data_Daily_Clustered_based_on_AI_SF_SI/Cluster_' + str(int(cluster)) + '/'
folder = 'CAMELS-US/Daily_Data/'

# Output folder, where we save the results
# ourputfolder = 'Output_USCA/outputs_with_seasonality_removed_and_p_and_pet_as_inputs/general_model_cluster_'       + str(int(cluster)) +
outputfolder = 'CAMELS-US/Output_USCA/'

if not os.path.exists(outputfolder):

    os.makedirs(outputfolder)
    print('Oops! directory did not exist, but no worries, I created it!')


SaveModel = outputfolder

#Static Data- it must contain items listed by "staticColumns" and grid code
path_static = 'CAMELS-US/Attributes//attributes.csv'

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

TraindGridCodes = np.array([])
iter = 1
for file in os.listdir(folder):
      if file.endswith(".csv"):

            GridCode = int(file.rstrip(".csv"))  # Extract GridCodes

            print('GridCode: ',GridCode)

            Dir = folder + str(file)

            df = pd.read_csv(Dir).dropna()
            df['date'] = pd.to_datetime(df.pop('date'))

            df['day_of_year'] = df['date'].dt.dayofyear

            # Apply log transformation on the target (log(x+1))
            df[TargetLabel] = np.log1p(df[TargetLabel])

            # Splitting Data (Train, Validation, Test)
            train_val_size = int(len(df) * TrainRatio)
            Val_size = int(train_val_size * ValidationRatio)

            train_val, test = df.iloc[0:train_val_size].copy(), df.iloc[train_val_size:].copy()
            train, val = train_val.iloc[0:(train_val_size-Val_size)].copy(), train_val.iloc[(train_val_size-Val_size):].copy()
      
            # Normalizing Input Data
            f_transformer = StandardScaler().fit(train[f_columns])

            train[f_columns] = f_transformer.transform(train[f_columns])
            val[f_columns] = f_transformer.transform(val[f_columns])
            test[f_columns] = f_transformer.transform(test[f_columns])
            train_val[f_columns] = f_transformer.transform(train_val[f_columns])


            static_row = dfs[dfs['gridcode'] == GridCode]
            for item in staticColumns:
              train.loc[:, item] = static_row[item].to_numpy()[0]
              train_val.loc[:, item] = static_row[item].to_numpy()[0]
              val.loc[:, item] = static_row[item].to_numpy()[0]
              test.loc[:, item] = static_row[item].to_numpy()[0]
          
            input_columns = f_columns + staticColumns

            X_train, y_train,train_date, train_days = create_dataset(train[input_columns],    train[TargetLabel],   train['date'],    train['day_of_year'],     time_steps=TIME_STEP)
            X_train_val, y_train_val,train_val_date, train_val_days = create_dataset(train_val[input_columns],  train_val[TargetLabel], train_val['date'], train_val['day_of_year'], time_steps=TIME_STEP)
            X_test, y_test, test_date,test_days = create_dataset(test[input_columns],  test[TargetLabel], test['date'],   test['day_of_year'], time_steps=TIME_STEP)
            X_val, y_val, val_date, val_days = create_dataset(val[input_columns], val[TargetLabel],val['date'], val['day_of_year'], time_steps=TIME_STEP)
            
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHs,
                batch_size=BatchSize,
                validation_data=(X_val, y_val),
                shuffle=True,
                callbacks=callbacks
              )


            TraindGridCodes = np.append(TraindGridCodes, GridCode)
            np.savetxt(SaveModel + 'Grodcodes_Based_On_Which_Trained_sofar.out', TraindGridCodes, delimiter=',')
            path = SaveModel + 'interationNum_' + str(int(iter))+'_Generally_Trained_UP_TO_NOW_Model' + '.h5'
            model.save_weights(path)
            os.remove(Dir)
            iter += 1
            # all_X_train.append(X_train)
            # all_y_train.append(y_train)
            
            # all_X_train_val.append(X_train_val)
            # all_y_train_val.append(y_train_val)
            
            # all_X_test.append(X_test)
            # all_y_test.append(y_test)
            
            # all_X_val.append(X_val)
            # all_y_val.append(y_val)

# Convert lists of arrays into single numpy arrays
# final_X_train = np.concatenate(all_X_train, axis=0)
# final_y_train = np.concatenate(all_y_train, axis=0)

# final_X_train_val = np.concatenate(all_X_train_val, axis=0)
# final_y_train_val = np.concatenate(all_y_train_val, axis=0)

# final_X_test = np.concatenate(all_X_test, axis=0)
# final_y_test = np.concatenate(all_y_test, axis=0)

# final_X_val = np.concatenate(all_X_val, axis=0)
# final_y_val = np.concatenate(all_y_val, axis=0)

          
# Model final outputs
path = SaveModel + 'Generally_Trained_Model' +'.h5'
model.save_weights(path)