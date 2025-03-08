#perbandingan model lain
# import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, MaxPooling1D
from tensorflow.keras.layers import Conv1D, Flatten, SimpleRNN
from keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# mengambil data dari excel
df = pd.read_csv('personal_transactions.csv')
# df.head()

# plot dan analysis data
df2 = df.reset_index()['Amount']
# plt.plot(df2)

# Calculate the threshold (e.g., 1.5*IQR for outliers)
Q1 = df2.quantile(0.25)
Q3 = df2.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Classify outliers as 'Unexpected Expense'
df['Category'] = np.where((df2 < lower_bound) | (df2 > upper_bound),
                           'Unexpected Expense', 'Regular Expense')

regular_expenses = df[df['Category'] == 'Regular Expense']
df2_regular = regular_expenses['Amount']

unexpected_expenses = df[df['Category'] == 'Unexpected Expense']
df2_unexpected = unexpected_expenses['Amount']

print(df2_regular)
print(df2_unexpected)

# data preprocessing
# 1. Log Transformation
df2_log = np.log(df2_regular + 1)  # Add 1 to avoid log(0)
print("Log transform :",df2_log)

# 3. Scale with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df2 = scaler.fit_transform(np.array(df2_log).reshape(-1,1))
print(df2)
print(df2.shape)

# membagi data training dan test
train_size = int(len(df2)*0.70)
test_size = len(df2) - train_size
train_data,test_data = df2[0:train_size,:],df2[train_size:len(df2),:1]
print("training data :", train_data)
print("testing data :", test_data)

# # membuat matriks dataset
def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)

# # panggil fungsi diatas
time_step = 1
X_train,Y_train =  create_dataset(train_data,time_step)
X_test,Y_test =  create_dataset(test_data,time_step)

X_train_other, Y_train_other = create_dataset(train_data, time_step=2)
X_test_other, Y_test_other = create_dataset(test_data, time_step=2)

# coba print hasil
print("X train :", X_train.shape)
print("Y train :", Y_train.shape)
print("X test :", X_test.shape)
print("Y test :", Y_test.shape)

class EarlyStopLogger(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:  # Checks if training stopped
            self.stopped_epoch = epoch + 1  # Epoch is 0-based, so add 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None:
            early_stopping_epochs[self.model_name] = self.stopped_epoch

# Dictionary untuk menyimpan epoch early stopping
early_stopping_epochs = {}

# Reshaping input to 3D for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train_other = X_train_other.reshape((X_train_other.shape[0], X_train_other.shape[1], 1))
X_test_other = X_test_other.reshape((X_test_other.shape[0], X_test_other.shape[1], 1))

X_train_mlp = X_train.reshape(X_train.shape[0], -1)
X_test_mlp = X_test.reshape(X_test.shape[0], -1)

# Model CNN
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', input_shape=(X_train_other.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
cnn_logger = EarlyStopLogger("CNN")
cnn_model.fit(X_train_other, Y_train_other, validation_data=(X_test_other, Y_test_other), epochs=200, batch_size=64, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True), cnn_logger])

# Model RNN
rnn_model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(X_train_other.shape[1], 1)),
    SimpleRNN(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
rnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
rnn_logger = EarlyStopLogger("RNN")
rnn_model.fit(X_train_other, Y_train_other, validation_data=(X_test_other, Y_test_other), epochs=200, batch_size=64, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True), rnn_logger])

# Model MLP
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_mlp.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
mlp_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
mlp_logger = EarlyStopLogger("MLP")
mlp_model.fit(X_train_mlp, Y_train, validation_data=(X_test_mlp, Y_test), epochs=200, batch_size=64, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True), mlp_logger])

# # buat lstm modelnya
# Defining the LSTM model with dropout for regularization
model = Sequential()

data_size = len(X_train)  # Get the size of the training data

# Set learning rate based on data size condition
if data_size <= 100:
    learning_rate = 0.001
else:
    learning_rate = 0.0001

# First Bidirectional LSTM layer
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(25, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

# liat hasil model
model.summary()
# masukan data traning dan test
lstm_logger = EarlyStopLogger("LSTM")
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 200,batch_size = 64,verbose = 1, callbacks=[early_stopping, lstm_logger])
# model.fit(X_train,Y_train,validation_data = (X_test,Y_test),epochs = 10,batch_size = 64,verbose = 1)

# lalu di prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print("train_predict :", train_predict)
print("test_predict :", test_predict)
# ubah ke bentuk original
train_predict = np.exp(scaler.inverse_transform(train_predict)) - 1
test_predict = np.exp(scaler.inverse_transform(test_predict)) - 1
actual_data = np.exp(scaler.inverse_transform(df2)) - 1

# Fungsi untuk prediksi iteratif
def predict_future_days(model, all_test_data, days_to_predict):

    # Initialize the list of predictions with all test data
    temp_input = list(all_test_data.flatten())  # Convert to list for easy manipulation
    future_predictions = []

    for _ in range(days_to_predict):
        x_input=np.array(temp_input[-1:])
        print("day input {}".format(x_input))
        # Prepare the current input using the last step
        current_input = np.array(temp_input[-1:]).reshape(1, 1, 1)  # Reshape for LSTM model
        next_prediction = model.predict(current_input, verbose=0)[0, 0]  # Predict next step

        # Append the prediction to the output list and extend temp_input
        future_predictions.append(next_prediction)
        temp_input.append(next_prediction)

    return np.array(future_predictions)

# Use the last 7 test data points for prediction
days_to_predict = 7  # or 30
all_test_data = test_data  # Use the entire scaled test data
future_predictions = predict_future_days(model, all_test_data, days_to_predict)

# Inverse transform predictions
future_predictions_original = np.exp(scaler.inverse_transform(future_predictions.reshape(-1, 1))) - 1

# Print and plot results
print("Test data:",test_data)
print(f"Predictions for the next {days_to_predict} days:", future_predictions_original.flatten())


# # print hasil plotnya
trainPredictPlot = np.empty_like(df2)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[time_step : len(train_predict)+time_step,:] = train_predict

testPredictPlot = np.empty_like(df2)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(time_step)*2 + 1 : len(df2) - 1,:] = test_predict

# plot
plt.figure(figsize=(14, 7)) #(width, height)
plt.plot(actual_data, label='Actual Data', color='blue')
plt.plot(train_predict, label='Train Predictions', color='green')
plt.plot(testPredictPlot, label='Test Predictions', color='orange')
plt.legend()
plt.show()

train_comparison_df = pd.DataFrame({
    'Actual (Train)': actual_data.flatten()[:20],
    'Predicted (Train)': train_predict.flatten()[:20]
})
pd.set_option('display.float_format', '{:,.0f}'.format)

# Display the DataFrame
print(train_comparison_df)

print("Train RMSE:", math.sqrt(mean_squared_error(Y_train, train_predict)))
print("Test RMSE:", math.sqrt(mean_squared_error(Y_test, test_predict)))


# Calculate MAE for train and test predictions
train_mae = mean_absolute_error(Y_train, train_predict)
test_mae = mean_absolute_error(Y_test, test_predict)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

# Evaluasi Model
def evaluate_model(model, X_test, y_test, reshape=False):
    if reshape:
        X_test = X_test.reshape(X_test.shape[0], -1)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return mae, rmse, pred

lstm_mae, lstm_rmse, _ = evaluate_model(model, X_test, Y_test)
cnn_mae, cnn_rmse, _ = evaluate_model(cnn_model, X_test_other, Y_test_other)
rnn_mae, rnn_rmse, _ = evaluate_model(rnn_model, X_test_other, Y_test_other)
mlp_mae, mlp_rmse, _ = evaluate_model(mlp_model, X_test_mlp, Y_test, reshape=True)

# Cetak hasil evaluasi
print("Evaluasi Model:")
print(f"LSTM  - MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}")
print(f"CNN   - MAE: {cnn_mae:.4f}, RMSE: {cnn_rmse:.4f}")
print(f"RNN   - MAE: {rnn_mae:.4f}, RMSE: {rnn_rmse:.4f}")
print(f"MLP   - MAE: {mlp_mae:.4f}, RMSE: {mlp_rmse:.4f}")

# Fungsi untuk mendapatkan detail layer
def get_model_layers_info(model, model_name):
    layer_info = []
    for i, layer in enumerate(model.layers):
        output_shape = layer.output.shape if hasattr(layer, 'output') else 'Unknown'
        layer_info.append({
            "Model": model_name,
            "Layer": i,
            "Type": layer.__class__.__name__,
            "Output Shape": str(output_shape)
        })
    return layer_info

# Mengumpulkan informasi dari semua model
models = {
    "CNN": cnn_model,
    "RNN": rnn_model,
    "MLP": mlp_model,
    "LSTM": model
}

all_layer_info = []
for name, mod in models.items():
    all_layer_info.extend(get_model_layers_info(mod, name))

# Membuat tabel dengan Pandas
df = pd.DataFrame(all_layer_info)

# Membuat tabel epoch early stopping
df_early_stopping = pd.DataFrame(list(early_stopping_epochs.items()), columns=["Model", "Early Stopping Epoch"])

# Menampilkan tabel
print("\nTabel model tiap layer:")
print(df.to_string(index=False))

print("\nEpoch saat Early Stopping:")
print(df_early_stopping.to_string(index=False))

# Function to print model summary and count parameters
def print_model_params(model, model_name):
    print(f"\nModel: {model_name}")
    model.summary()  # Print model architecture
    total_params = model.count_params()  # Get total trainable parameters
    print(f"Total Trainable Parameters for {model_name}: {total_params}\n")
    return total_params

# Count parameters for CNN model
cnn_params = print_model_params(cnn_model, "CNN")

# Count parameters for RNN model
rnn_params = print_model_params(rnn_model, "RNN")

# Count parameters for MLP model
mlp_params = print_model_params(mlp_model, "MLP")

# Count parameters for LSTM model
lstm_params = print_model_params(model, "LSTM")

# Print all results in a structured format
import pandas as pd
params = pd.DataFrame({
    "Model": ["CNN", "RNN", "MLP", "LSTM"],
    "Total Parameters": [cnn_params, rnn_params, mlp_params, lstm_params]
})
print(params)