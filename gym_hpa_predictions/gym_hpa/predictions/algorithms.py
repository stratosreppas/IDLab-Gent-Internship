import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras
import plotly.graph_objs as go
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


class Prediction:


    def __init__(self, date_col, val_col, trainset, testset=None, current_value=1, n_steps=1, past_horizon = 60):
        """
        This class contains all the prediction algorithms used in the project, as well as all the functions needed
        to modify the dataset if necessary to make accurate predictions

        :param date_col: The name of the date column
        :param val_col: The name column that you want to predict without the deployment name (e.g. _cpu_usage)
        :param trainset: The directory of the dataset that you will use for training certain models
        :param testset: The directory of the dataset that you will use for testing. If not defined, it is the same as
                        the trainset.
        :param current_value: The value that the metric has right now (probably not saved in the dataset yet)
        :param n_steps: The forecasting horizon
        :param past_horizon: The number of past observations that the model has access to, to make the predictions
        """

        # super(Prediction, self).__init__()
        self.trainset = pd.read_csv(trainset)

        self.testset = pd.read_csv(testset) if testset is not None else  \
            pd.read_csv(trainset)

        self.date_col = date_col
        self.val_col = val_col
        self.current_value = current_value
        self.n_steps = n_steps
        self.past_horizon = past_horizon


    def convert_to_timeseries(self):
        """
         Because the datasets might not be equal spaced (dublicate observations in a second, no observations
         in a second, missing values), this function converts them to a timeseries so that every second there is exactly
         one observation. The missing values are filled as the value of the previous observation.

        :return: the final timeseries
        """
        datasets = [self.trainset, self.testset]

        # Step 0: Drop duplicates
        for dataset in datasets:
            dataset.drop_duplicates(subset=[self.date_col], keep='first', inplace=True)

        df_resampled = []

        for i in range(len(datasets)):
            df = datasets[i].copy()  # Create a copy of the DataFrame to avoid any potential side-effects

            # Step 1: Convert Timestamp column to pandas DateTime type
            df[self.date_col] = pd.to_datetime(df[self.date_col])

            # Step 2: Set the Timestamp column as the index
            df.set_index(self.date_col, inplace=True)

            # Step 3: Resample and create a complete time range with desired frequency (e.g. 1 second)
            frequency = "1S"
            df_resampled.append(df.resample(frequency).asfreq())

            # If you want to fill the missing values with a specific value, you can do the following:
            # value =
            # df_resampled.fillna(value, inplace=True)

            # If you want to fill the missing values using forward fill (carrying the last known value forward):
            df_resampled[i].fillna(method='ffill', inplace=True)

            # If you want to fill the missing values using backward fill (carrying the next known value backward):
            # df_resampled.fillna(method='bfill', inplace=True)

            # Step 4: Reset the index if needed
            df_resampled[i].reset_index(inplace=True)

        return df_resampled

    @staticmethod
    def load_lstm():
        #loaded_model = tf.keras.models.load_model(prediction/models/lstm)
        return

    def create_env_for_NN(self, data, new_value=None):
        """

        :param data: the dataset to be converted
        :param new_value: the latest value, that is not yet saved in the dataset. It is used for the out-of-sample lstm.
        :return: x: the scaled observation space for the lstm (past observations)
                 y: the actual values (used for error correction and learning purposes)
                 scaler: the scaler used to scale the dataset for training. It will then be used to convert the scaled predictions to actual values
        """
        dataset = data.filter(['redis-predictions']).values

        if new_value is not None:
            a_list = dataset.tolist()

            # Append the new value to the list
            a_list.append([new_value])

            # Convert the list back to a NumPy array
            dataset = np.array(a_list)
            print(dataset)

        env_data_len = int(np.ceil(len(dataset)))

        # Scale our data from 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Use our scaled data for training
        env_data = scaled_data[0:int(env_data_len), :]
        x = []
        y = []

        #y.append(env_data[0:50, 0])
        for i in range(self.past_horizon, len(env_data)):
            x.append(env_data[i - self.past_horizon:i, 0])
            y.append(env_data[i, 0])

        x, y = np.array(x), np.array(y)

        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        return x, y, scaler

    def moving_average_n(self, n):
        """
        Take the average of the testset and trainset every n seconds and return the result
        :param n:
        :return:
        """
        testset = self.testset.iloc[::n, :]
        trainset = self.testset.iloc[::n, :]
        return trainset, testset

    @staticmethod
    def lstm_train(x_train, y_train, batch_size, epochs):

        # Build LSTM model

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.35))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        # Compile the model. We use MSE because it penaltizes big errors, which is what we want to avoid
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Тrain the model
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        model.save('prediction/models/lstm')
        # Structure of the model
        #keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

        return model

    def lstm_test(self, model, x_test, scaler):
        # Predict on test data
        lstm_predictions = model.predict(x_test)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)
        lstm_predictions = np.vstack((self.testset.filter(['redis-leader_cpu_usage']).values[0:60].astype(np.int), lstm_predictions.astype(np.int)))
        return lstm_predictions

    def naive(self):
        """
        Implementation of the naive method in-sample. The forecast for the next value is the current one. It is used as
        a benchmark. In very random environments, it often outperforms more complex models.

        :return: the predictions for every value of the dataset
        """
        testset = self.testset.filter([self.val_col]).values
        naive_predictions = np.roll(testset, 1)
        naive_predictions[0] = naive_predictions[1]
        return naive_predictions

    def naive_dynamic(self):
        """
        Implementation of the naive method out-of-sample. The forecast for the next value is the current one. It is used as
        a benchmark. In very random environments, it often outperforms more complex models.

        :return: the prediction for the next value
        """
        return self.testset.filter([self.val_col]).values[-1]

    def ses(self, alpha):
        """
        Implementation of the SES (Simple Exponential Smoothing) algorithm in-sample. The prediction is used

        :param alpha:
        :return:
        """
        timeseries = self.testset.filter([self.val_col]).values
        forecast = []
        forecast.append(int(np.mean(timeseries)))  # calculate the average to be the starting value for the forecast

        for t in range(len(timeseries)):
            forecast.append(int(alpha * timeseries[t] + (1 - alpha) * forecast[t]))
        return forecast[:-1]

    def ses_dynamic(self, alpha=0.4):
        forecast = int(alpha * self.current_value + (1 - alpha) * self.testset.filter([self.val_col+'_predictions']).values[-1])
        return forecast

    def arima_fit(self):
        testset = self.testset[self.val_col].values
        stepwise_fit = auto_arima(testset, start_p=0, start_d =0, start_q=0,
                          max_p=5, max_d=5, max_q=5, seasonal=False, trace = False,
                          supress_warnings=True)
        p, d, q = stepwise_fit.order
        # Fit ARIMA model with optimal parameters
        model = ARIMA(testset, order=(p, d, q))
        fitted_model = model.fit()
        return fitted_model

    def arima(self, model):
        forecast = model.predict()
        return forecast

    def arima_dynamic(self, model):
        forecast = model.forecast(steps=self.n_steps)

        return forecast


    def lstm(self, batch_size, epochs):
        x_train, y_train, scaler_train = self.create_env_for_NN(self.trainset)
        x_test, y_test, scaler_test = self.create_env_for_NN(self.testset)
        print('Model training...')
        model = self.lstm_train(x_train, y_train, batch_size, epochs)
        print('Model testing...')
        lstm_predictions = self.lstm_test(model, x_test, scaler_test)
        return lstm_predictions

    def lstm_test_dynamic(self, batch_size, epochs, model, observation):
        x_test, y_test, scaler_test = self.create_env_for_NN(self.testset)
        lstm_predictions = self.lstm_test(model, x_test, scaler_test)
        return lstm_predictions[-1]

    def average(self, prediction: dict[str, list] = None):
        """
        Implementation of the average of many independent prediction algorithms in-sample.
        It is a more moderate approach, usually wielding more accurate results.

        :param prediction: optional - a dictionary containing the predictions formed by the algorithms.
                           The key must be a string and the value a list containing floats.
                           If not provided, the naive/ses(0.4)/arima are used. You can change them in the code.
        :return: the predictions for every value of the dataset
        """
        if prediction is None:
            prediction = {}

            prediction['Naive'] = self.naive()
            prediction['SES'] = self.ses(0.4)
            # add more if you want to

        return np.mean(prediction.values(), axis=0)

    def average_dynamic(self, prediction: dict[str, float] = None):
        """
        Implementation of the average of many independent prediction algorithms out-of-sample.
        It is a more moderate approach, usually wielding more accurate results.

        :param prediction: optional - a dictionary containing the predictions formed by the algorithms.
                           The key must be a string and the value a float.
                           If not provided, the naive/ses(0.4)/arima are used. You can change them in the code.
        :return: the predictions for every value of the dataset
        """
        if prediction is not None:
            prediction = {}

            prediction['Naive'] = self.naive_dynamic()
            prediction['SES'] = self.ses_dynamic(0.4)
            # add more if you want to

        return np.mean(prediction.values(), axis=0)

    def rmse(self):
        """
        Calculation of the RMSE (Root Mean Squared Error) for the predictions of the testset.
        The lower the value, the higher the accuracy.
        It is expressed in the same units as the values in the dataset.
        Gives no information about the bias of the algorithm.
        It emphasises larger errors.

        :return: The calculated RMSE
        """
        RMSE = np.sqrt(np.mean(((self.testset[self.val_col+'_predictions'] - self.testset[self.val_col]) ** 2)))

        return RMSE

    def mae(self):
        """
        Calculation of the MAE (Mean Absolute Error) for the predictions of the testset.
        The lower the value, the higher the accuracy.
        It is expressed in the same units as the values in the dataset.
        Gives no information about the bias of the algorithm.

        :return: The calculated MAE
        """
        MAE = np.mean(np.abs(self.testset[self.val_col] - self.testset[self.val_col+'_predictions']))
        return MAE

    def mape(self):
        """
        Calculation of the MAPE for the predictions of the testset.
        The lower the value, the higher the accuracy.
        It is expressed in a percentage (%).
        Gives no information about the bias of the algorithm. It is the normalized version of the MAE.
        Because it is normalized it is more suitable for comparing the accuracy of an algorithm in different datasets.

        :return: The calculated MAPE
        """
        MAPE = np.mean(np.abs((self.testset[self.val_col] - self.testset[self.val_col+'_predictions']) / self.testset[self.val_col])) * 100
        return MAPE

    def me(self):
        """
        Calculation of the ME (Mean Error) for the predictions of the testset.
        The lower the value, the higher the accuracy.
        It is expressed in the same units as the values in the dataset.
        Negative values indicate that the algorithm tends to overshoot (predictions usually higher than the real values)
        Positive values indicate that the algorithm tends to undershoot (predictions usually lower than the real values)

        :return: The calculated ME
        """
        ME = np.mean(self.testset[self.val_col] - self.testset[self.val_col + '_predictions'])
        return ME


