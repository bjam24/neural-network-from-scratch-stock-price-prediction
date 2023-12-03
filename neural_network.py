import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class ArtificialNeuralNetwork:
    def __init__(self, input_layer_neurons, hidden_layer_neurons, output_layer_neurons):
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons

        self.weights_1 = np.random.randn(input_layer_neurons, hidden_layer_neurons)
        self.weights_2 = np.random.randn(hidden_layer_neurons, output_layer_neurons)

    def sigmoid(self, Z): # Activation function
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.weights_1)
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights_2)
        A2 = self.sigmoid(self.Z2)
        return A2

    def back_propagation(self, A2, X, Y, learning_rate):
        output_layer_error = A2 - Y[:,None]
        delta_output_error = output_layer_error * A2 * (1 - A2)

        hidden_layer_error = np.dot(delta_output_error, self.weights_2.T)
        delta_hidden_error = hidden_layer_error * self.A1 * (1 - self.A1)

        self.weights_2 = self.weights_2 - learning_rate * np.dot(self.A1.T, delta_output_error) / Y.size
        self.weights_1 = self.weights_1 - learning_rate * np.dot(X.T, delta_hidden_error) / Y.size

    def train(self, x_train, y_train, learning_rate, iterations):
        for i in range(iterations):
            output = self.forward_propagation(x_train)
            self.back_propagation(output, x_train, y_train, learning_rate)

    def predict(self, X):
        return self.forward_propagation(X)


if __name__ == "__main__":
    # Loading stock data
    stock_data = pd.read_csv('data/TSLA.csv')

    # Preparation of dataset
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['Open-Close'] = pd.DataFrame(stock_data['Open'] - stock_data['Close'])
    stock_data['7 Days MA'] = stock_data['Close'].rolling(7).mean()
    stock_data['14 Days MA'] = stock_data['Close'].rolling(14).mean()
    stock_data['21 Days MA'] = stock_data['Close'].rolling(21).mean()
    stock_data['7 Days Std Dev'] = stock_data['Close'].rolling(7).std()

    features = ['High-Low', 'Open-Close', '7 Days MA', '14 Days MA', '21 Days MA', '7 Days Std Dev']
    dataset = stock_data[features + ['Close']].dropna().reset_index(drop=True)

    # Scaling dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_2 = MinMaxScaler(feature_range=(0, 1)) # The second scaler is used to resale predicted data
    scaled_dataset = scaler.fit_transform(dataset)
    scaled_dataset_2 = scaler_2.fit_transform(dataset['Close'].values.reshape(-1, 1))

    # Split dataset into features and target
    x_scaled_dataset = scaled_dataset[:, [0, 1, 2, 3, 4, 5]]
    y_scaled_dataset = scaled_dataset[:, 6]

    # Split features and target set into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled_dataset, y_scaled_dataset, random_state=104,
                                                        test_size=0.20, shuffle=True)

    # ANN initialization, training and prediction
    ANN = ArtificialNeuralNetwork(6, 3, 1)
    ANN.train(x_train, y_train, 0.1, 400000)
    y_predicted = ANN.predict(x_scaled_dataset)

    # Preparing new dataset for visualization
    y_predicted = scaler_2.inverse_transform(y_predicted) # rescaling predicted data
    visual_dataset = stock_data[['Date', 'Close']].iloc[20:]
    visual_dataset['Date'] = pd.to_datetime(visual_dataset['Date'])
    visual_dataset['Predicted'] = y_predicted

    # Data visualization
    plt.plot(visual_dataset['Date'], visual_dataset['Close'], '-s', linewidth=0.7, markersize=4, color='darkblue',
             label='Original Closing Price')
    plt.plot(visual_dataset['Date'], visual_dataset['Predicted'], '-o', linewidth=0.7, markersize=4, color='red',
             label='Predicted Closing Price')
    plt.title('TSLA Close and Predicted Prices')
    plt.legend(loc="upper left", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.show()