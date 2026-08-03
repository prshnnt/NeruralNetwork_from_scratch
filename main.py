import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("./sample_data/mnist_train_small.csv", header=None)
test = pd.read_csv("./sample_data/mnist_test.csv", header=None)

lr = 0.001
EPOCHS = 5

# train.iloc[:,1:] /= 255.0

def get_at(idx,data):
	return data.iloc[idx, 1:]
def get_label_at(idx,data):
	return data.iloc[idx, 0]

def one_hot(label):
	#one hot encoded Y vector
	y = np.zeros((10,1))
	y[label] = 1
	return y

def display(idx,data):
	plt.imshow(np.array(get_at(idx,data)).reshape(28, 28), cmap="gray")
	plt.show()

# print(set(get_at(0)))
# print(get_at(0).shape)
# display(0)

def relu(Z):
	return np.maximum(0, Z)

def relu_derivative(Z):
	return Z > 0

def softmax(z):
		z = z - np.max(z, axis=0, keepdims=True)
		e = np.exp(z)
		return e / np.sum(e, axis=0, keepdims=True)


class Layer:
	def __init__(self,ip,op,prev_layer=None,last=False):
		self.X = None
		self.W = np.random.randn(op, ip) * np.sqrt(2.0 / ip)
		self.b = np.zeros((op,1))
		self.Z = None
		self.A = None
		self.prev_layer = prev_layer
		self.last = last

	def forward(self, X):
		self.X = X.reshape(-1, 1)
		self.Z = self.W @ self.X + self.b
		# print("Z shape:", self.Z.shape)
		self.A = relu(self.Z) if not self.last else softmax(self.Z)
		return self.A

	def backward(self,delta):
		dW = delta @ self.X.T
		db = delta
		# calc delta before updating weights
		if self.prev_layer is not None:
			delta = (self.W.T @ delta) * relu_derivative(self.prev_layer.Z)
		self.W -= lr * dW
		self.b -= lr * db
		if self.prev_layer is not None:
			return self.prev_layer.backward(delta)

class Model:
	def __init__(self,shape):
		self.layers = []
		for i in range(len(shape)-1):
			self.layers.append(
					Layer(
							shape[i],shape[i+1],
							prev_layer=(
									self.layers[-1] if len(self.layers)>0 else None
									)
							)
					)
		self.layers[-1].last = True
	def forward(self,input):
		for layer in self.layers:
			input = layer.forward(input)
		return input
	def backward(self,delta):
		self.layers[-1].backward(delta)
	def fit(self,X,Y):
		for x,y in zip(X,Y):
			y_hat = self.forward(x)
			delta = y_hat - y
			self.backward(delta)
	def predict(self,x):
		return np.argmax(self.forward(x))


def train_model(model):
    # Normalize pixel values to [0, 1] -- fixes exploding activations/gradients
    X_all = np.array(train.iloc[:, 1:]) / 255.0
    Y_all = np.array([one_hot(label) for label in train.iloc[:, 0]])
 
    for epoch in range(EPOCHS):
        permutation = np.random.permutation(len(train))
        X = X_all[permutation]
        Y = Y_all[permutation]
        model.fit(X, Y)
 
        # quick training accuracy check
        correct = 0
        sample_n = min(2000, len(X_all))
        for i in range(sample_n):
            pred = model.predict(X_all[i])
            label = np.argmax(Y_all[i])
            correct += (pred == label)
        print(f"epoch: {epoch}  train acc (sample): {correct / sample_n:.4f}")
 
 
def test_model(model):
    i = random.randint(0, len(test) - 1)
    x = np.array(get_at(i, test)) / 255.0  # normalize test input too
    print("Predicted:", model.predict(x))
    print("Actual:", get_label_at(i, test))
    display(i, test)


def main():
	model = Model([784,128,10])
	train_model(model)
	test_model(model)

if __name__ == "__main__":
	main()