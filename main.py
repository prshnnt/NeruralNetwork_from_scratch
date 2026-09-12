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
batch_size = 32

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
def cross_entropy(Y, Y_hat):
    eps = 1e-12
    return -np.sum(Y * np.log(Y_hat + eps)) / Y.shape[1]

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
    def __init__(self,ip,op,last=False):
        self.X = np.zeros((ip,1))
        self.W = np.random.randn(op, ip) * np.sqrt(2.0 / ip)
        self.b = np.zeros((op,1))
        self.Z = None
        self.A = None
        self.grad = Grad(np.zeros((op,ip)),np.zeros((op,1)))
        self.momentum = Momentum(np.zeros((op,ip)),np.zeros((op,1)))
        self.last = last

    def forward(self, X):
        # self.X = X.reshape(-1, 1)
        self.X = X
        self.Z = self.W @ self.X + self.b
        if self.last:
            self.A = softmax(self.Z)
        else:
            self.A = relu(self.Z)
        return self.A

class Model:
    def __init__(self,shape,optimizer="sgd"):
        self.layers:List[Layer] = []
        self.optimizer = optimizer
        for i in range(len(shape)-1):
            self.layers.append(
                    Layer(shape[i],shape[i+1], last=(i == len(shape) - 2))
                    )
    def forward(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    def backward(self,delta):

        batch_size = delta.shape[1]
        
        for i in reversed(range(len(self.layers))):

            layer = self.layers[i]

            dW = delta @ layer.X.T / batch_size
            db = np.sum(delta,axis=1,keepdims=True) / batch_size
            layer.grad.set(dW,db)
            if i>0:
                delta = (
                    layer.W.T @ delta
                    * relu_derivative(self.layers[i-1].Z)
                )
            
    def fit(self,X,Y,batch_size = 64):
        n = X.shape[1] # len of input data
        total_loss = 0.0

        for start in range(0, n , batch_size):
            end = min(start + batch_size,n)

            X_batch = X[:,start:end]
            Y_batch = Y[:,start:end]

            # forward pass
            Y_hat = self.forward(X_batch)

            total_loss += cross_entropy(
                Y_batch,
                Y_hat
            ) * X_batch.shape[1]

            delta = Y_hat - Y_batch

            self.backward(delta)

            self.optimize()
        return total_loss / n
    
    def predict(self,x):
        return np.argmax(self.forward(x),axis=0)
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
                layer.momentum.mW = beta*layer.momentum.mW + (1. - beta) * layer.grad.dW
                layer.momentum.mb = beta*layer.momentum.mb + (1. - beta) * layer.grad.db
                layer.W -= lr * layer.momentum.mW
                layer.b -= lr * layer.momentum.mb

def train_model(model:Model):
    # Normalize pixel values to [0, 1] -- fixes exploding activations/gradients
    X_all = train.iloc[:, 1:].to_numpy(dtype=np.float32).T / 255.0

    labels = train.iloc[:,0].to_numpy(dtype=int)

    Y_all = np.zeros((10,len(labels)),dtype=int)
    Y_all[labels,np.arange(len(labels))]=1
 
    for epoch in range(EPOCHS):
        permutation = np.random.permutation(X_all.shape[1])
        X = X_all[:,permutation]
        Y = Y_all[:,permutation]
        loss = model.fit(X, Y,batch_size)

        print(f"epoch: {epoch}  loss:{loss:.5f}")
 
 
def test_model(model:Model):
    X_test = test.iloc[:,1:].to_numpy(dtype=np.float32).T / 255.0
    Y_test = test.iloc[:,0].to_numpy(dtype=int)

    pred = model.predict(X_test)

    accuracy = np.mean(pred==Y_test)

    print(f"Test Accuracy: {accuracy:.4f}")

    # testing the model and showing results
    i = random.randint(0, len(test) - 1)
    x = X_test[:,i:i+1]
    print("Predicted:", model.predict(x)[0])
    print("Actual:", Y_test[i])
    display(i, test)


def main():
    model = Model([784,128,10])
    train_model(model)
    test_model(model)

if __name__ == "__main__":
    main()