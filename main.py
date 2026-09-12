import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

train = pd.read_csv("./sample_data/mnist_train_small.csv", header=None)
test = pd.read_csv("./sample_data/mnist_test.csv", header=None)

lr = 0.01
beta = .9
EPOCHS = 5

# train.iloc[:,1:] /= 255.0

def get_at(idx,data):
	return data.iloc[idx, 1:].to_numpy(dtype=np.float64)
def get_label_at(idx,data):
	return int(data.iloc[idx, 0])

def one_hot(label:int):
	#one hot encoded Y vector
	y = np.zeros((10,1))
	y[int(label)] = 1
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
    return (Z > 0).astype(float)

def softmax(z):
		z = z - np.max(z, axis=0, keepdims=True)
		e = np.exp(z)
		return e / np.sum(e, axis=0, keepdims=True)
def cross_entropy(y, y_hat):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return -np.sum(y * np.log(y_hat))

class Grad:
	def __init__(self,dW:np.ndarray,db:np.ndarray):		
		self.dW = dW
		self.db = db
	def set(self,dW,db):
		self.dW = dW
		self.db = db
class Momentum:
	def __init__(self,mW:np.ndarray,mb:np.ndarray):		
		self.mW = mW
		self.mb = mb
class Layer:
	def __init__(self,ip,op,prev_layer=None,last=False):
		self.X = np.zeros((ip,1))
		self.W = np.random.randn(op, ip) * np.sqrt(2.0 / ip)
		self.b = np.zeros((op,1))
		self.Z = None
		self.A = None
		self.grad = Grad(np.zeros((op,ip)),np.zeros((op,1)))
		self.momentum = Momentum(np.zeros((op,ip)),np.zeros((op,1)))
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
		self.grad.set(dW,db)
		if self.prev_layer is not None:
			return self.prev_layer.backward(delta)

class Model:
	def __init__(self,shape,optimizer="sgd"):
		self.layers:List[Layer] = []
		self.optimizer = optimizer
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
		total_loss = 0
		for x,y in zip(X,Y):
			y_hat = self.forward(x)
			total_loss += cross_entropy(y,y_hat)
			delta = y_hat - y
			self.backward(delta)
			self.optimize()
		return total_loss/len(X)
	def predict(self,x):
		return np.argmax(self.forward(x))
	def save(self):
		import pickle
		with open("model.pickle","wb") as f:
			pickle.dump(self,f)
	@staticmethod
	def load(path):
		with open(path,"rb") as f:
			import pickle
			obj = pickle.load(f)
			return obj
	def optimize(self):
		'''
        Stochatic Gradient Descent (SGD):
        θ^(t+1) <- θ^t - η∇L(y, ŷ)

        Momentum:
        v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
        θ^(t+1) <- θ^t - ηv^(t+1)
    	'''
		if self.optimizer == "sgd":
			for layer in self.layers:
				layer.W -= lr * layer.grad.dW
				layer.b -= lr * layer.grad.db
		elif self.optimizer == "momentum":
			for layer in self.layers:
				layer.momentum.mW = beta*layer.momentum.mW + layer.grad.dW
				layer.momentum.mb = beta*layer.momentum.mb + layer.grad.db
				layer.W -= lr * layer.momentum.mW
				layer.b -= lr * layer.momentum.mb

def train_model(model:Model):
    # Normalize pixel values to [0, 1] -- fixes exploding activations/gradients
    X_all = np.array(train.iloc[:, 1:]) / 255.0
    Y_all = np.array([one_hot(label) for label in train.iloc[:, 0]])
 
    for epoch in range(EPOCHS):
        permutation = np.random.permutation(len(train))
        X = X_all[permutation]
        Y = Y_all[permutation]
        loss = model.fit(X, Y)
 
        # quick training accuracy check
        correct = 0
        sample_n = min(2000, len(X_all))
        for x , label in zip(X_all,Y_all):
            pred = model.predict(x)
            correct += (pred == label)
        print(f"epoch: {epoch}  train acc (sample): {correct / sample_n:.4f}  loss:{loss:.5f}")
 
 
def test_model(model:Model):
	X_test = np.array(test.iloc[:,1:] / 255.0)
	Y_test = np.array(test.iloc[:,0],dtype=int)

	correct = 0
	for x , label in zip(X_test,Y_test):
		pred = model.predict(x)
		correct += (pred == label)

	accuracy = correct / len(X_test)

	print(f"Test Accuracy: {accuracy:.4f}")

	# testing the model and showing results
	i = random.randint(0, len(test) - 1)
	x = X_test[i]
	print("Predicted:", model.predict(x))
	print("Actual:", get_label_at(i, test))
	display(i, test)


def main():
	model = Model([784,128,10])
	train_model(model)
	test_model(model)

if __name__ == "__main__":
	main()