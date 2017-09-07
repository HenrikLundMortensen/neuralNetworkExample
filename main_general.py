import numpy as np
import sklearn.datasets
from matplotlib import pyplot as plt


X, y = sklearn.datasets.make_moons(200, noise=0.2)

ytmp = [None]*len(y)
for i in range(len(y)):
    if y[i] == 0:
        ytmp[i] = [1,0]
    else:
        ytmp[i] = [0,1]        
        

y = np.array(ytmp)
    

nn_hdim = 3
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2
nn_nlayer = 1

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


    # print('z1.shape = %s' %(z1.shape,))
    # print('a1.shape = %s' %(a1.shape,))

    correct_logprobs = -np.log(a2[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)
    return data_loss/num_examples

def getGradient(model,X,yhat,y,z):
    """

    """
    N = num_examples
    W = model[0]
    b = model[1]

    dLdb = [None]*(nn_nlayer+1)
    dLdW = [None]*(nn_nlayer+1)

    dLdb[nn_nlayer] = np.sum((yhat-y)/N,axis=0)
    dLdW[nn_nlayer] = np.outer(z[nn_nlayer-1][0],dLdb[nn_nlayer][0])

    i = nn_nlayer
    while i > 0:
        i -= 1
        dLdb[i] = np.sum(dLdb[i+1].dot(np.transpose(W[i+1])) * (1-z[i]*z[i]),axis=0)
        if i != 0:
            dLdW[i] = np.sum(np.outer(z[i-1],dLdb[i]),axis=0)
        else:
            dLdW[i] = np.sum(np.outer(X,dLdb[i]),axis=0)

    return [dLdW,dLdb]

def predict(model,x):
    """
    """
    W = model[0]
    b = model[1]

    z = [None]*nn_nlayer
    a = [None]*(nn_nlayer+1)

    for i in range(nn_nlayer+1):
        if i == 0:
            a[i] = np.dot(x,W[i])+ b[i]
            z[i] = np.tanh(a[i])
            # z.append(np.tanh(a[i]))
        else:
            if i != nn_nlayer:
                a[i] = np.dot(np.transpose(z[i-1]),W[i]) + b[i]
                # z.append(np.tanh(a[i]))
                z[i] = np.tanh(a[i])
            else:
                a[i] = np.dot(z[i-1],W[i]) + b[i]

    return [softmax(a[-1]),z]


def plotDecisionBoundary(model):
    """

    """
    # Create mesh grid

    xmin = min(X[:,0])-0.5
    xmax = max(X[:,0])+0.5
    ymin = min(X[:,1])-0.5
    ymax = max(X[:,1])+0.5

    h = 0.01
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

    Z = np.argmax(predict(model,np.c_[xx.ravel(),yy.ravel()])[0],axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,cmap = plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)



def build_model(nn_hdim,num_passes=1):
    """
    """
    eps = 0.01

    W = []
    b = []
    for i in range(nn_nlayer+1):
        if i == 0:
            W.append(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
            b.append(np.zeros((1, nn_hdim)))
        else:
            if i == nn_nlayer:
                W.append(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_input_dim))
                b.append(np.zeros((1, nn_output_dim)))
            else:
                W.append(np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim))
                b.append(np.zeros((1, nn_hdim)))

    model = [W,b]
    for i in range(num_passes):
        [yhat,z] = predict(model,X)
        yhat = np.round(yhat)
        [dLdW,dLdb] = getGradient(model,X,yhat,y,z)


        for i in range(nn_nlayer+1):
            W[i] += -eps*dLdW[i]
            b[i] += -eps*dLdb[i]

        # print('%.3f \t%i' %(calculateLoss([W1,b1,W2,b2]),i))
        model = [W,b]

    return model

model = build_model(3,num_passes=1000)

X, y = sklearn.datasets.make_moons(200, noise=0.3)
plotDecisionBoundary(model)
plt.savefig('decisionBoundary.png')
