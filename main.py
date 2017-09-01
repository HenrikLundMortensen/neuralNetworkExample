import numpy as np
import sklearn.datasets
from matplotlib import pyplot as plt


X, y = sklearn.datasets.make_moons(200, noise=0.2)

nn_hdim = 3
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

def softmax(xlist):
    """
    """

    res = [np.exp(x) for x in xlist]
    res /= np.sum(res,axis=1, keepdims=True)
    return res

    
def calculateLoss(model):
    """
    """

    W1 = model[0]
    b1 = model[1]
    W2 = model[2]
    b2 = model[3]    

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)

    correct_logprobs = -np.log(a2[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)
    return data_loss/num_examples


def predict(model,x):
    """
    """
    W1 = model[0]
    b1 = model[1]
    W2 = model[2]
    b2 = model[3]    

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)

    return np.argmax(a2,axis=1)
    


def plotDecisionBoundary(model):
    """

    """
    W1 = model[0]
    b1 = model[1]
    W2 = model[2]
    b2 = model[3]

    # Create mesh grid

    xmin = min(X[:,0])-0.5
    xmax = max(X[:,0])+0.5
    ymin = min(X[:,1])-0.5
    ymax = max(X[:,1])+0.5

    h = 0.01
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

    Z = predict(model,np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,cmap = plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

    


def build_model(nn_hdim,num_passes=10000):
    """
    """
    eps = 0.01
    
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    model = {}
    for i in range(num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)

        delta3 = a2
        delta3[range(num_examples),y] -=1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3,axis = 0,keepdims = True)
        delta2 = delta3.dot(W2.T)* (1 - np.power(a1,2))
        dW1 = np.dot(X.T,delta2)
        db1 = np.sum(delta2,axis=0)

        W1 += -eps*dW1
        b1 += -eps*db1         
        W2 += -eps*dW2
        b2 += -eps*db2
        print('%.3f \t%i' %(calculateLoss([W1,b1,W2,b2]),i))

    model = [W1,b1,W2,b2]
    return model

model = build_model(3,num_passes=10000)

plotDecisionBoundary(model)
plt.savefig('decisionBoundary.png')
