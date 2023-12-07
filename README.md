# Neural network from scratch stock price prediction
The purpose of this project is implementation of neural network from scratch, which can be used for stock price prediction.
To achieve this goal ANN is created with features listed below:
1. Stock High minus Low price (H-L)
2. Stock Close minus Open price (O-C)
3. Stock price’s seven days’ moving average (7 DAYS MA)
4. Stock price’s fourteen days’ moving average (14 DAYS MA)
5. Stock price’s twenty one days’ moving average (21 DAYS MA)
6. Stock price’s standard deviation for the past seven days (7 DAYS STD DEV)

**Architecture of neural network**

![image](https://github.com/bjam24/neural-network-from-scratch-stock-price-prediction/assets/61807667/cf2b3637-02fb-4848-b5f6-1ccf6877f272)

Described neural network is an example of Multilayer Perceptron (MLP). This method belongs to Supervised Learning and uses
backpropagation during training. Input layer includes 6 neurons for 6 features. There is only 1 neuron in output layer for
predicted price.
On the beginning of a project a dataset is created. It contains 6 mentioned features 'X' and Close Price which is perceived
as target 'Y'. After having scaled data, a dataset is split into training and testing sets. Training set X and Y are used for
changing neural network's weights during training. After this predicted Y (Closing price) is calculate. The user can choose
iterations and learning rate. The result of these doings is presented below.

**Visualization of predicted closing price in comparison to historical closing price**

![image](https://github.com/bjam24/neural-network-from-scratch-stock-price-prediction/assets/61807667/19261f38-c835-47c6-a343-cf32f52a899a)

The last part of this project is implementation of some accuracy measures froms scratch such as:
- RMSE
- MAPE
- MBE

## Data source
- https://finance.yahoo.com/
