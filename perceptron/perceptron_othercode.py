import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import joblib from matplotlib.colors 
import ListedColormap 
plt.style.use("fivethirtyeight")

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # RANDOM WEIGHT ASSIGNMENT
    print(f"initial weights before training: n{self.weights}")
    self.eta = eta # LEARNING RATE
    self.epochs = epochs 


  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z = W * X
    return np.where(z > 0, 1, 0) # ACTIVATION FUNCTION

  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # HERE WE ARE USING BIAS AS WELL
    print(f"X with bias: n{X_with_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias, self.weights) # forward pass 
      print(f"predicted value after forward pass: n{y_hat}")
      self.error = self.y - y_hat
      print(f"error: n{self.error}")
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) 
# backward propagation
      print(f"updated weights after epoch:n{epoch}/{self.epochs} : n{self.weights}")
      print("#####"*10)


  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_with_bias, self.weights)#Prediction function

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss: {total_loss}")
    return total_loss

def prepare_data(df):
    X=df.drop("y",axis=1)
    y=df["y"]
    return X,y
AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)# Calling the function

model.total_loss()
