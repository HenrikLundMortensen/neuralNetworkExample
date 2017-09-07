import numpy as np
import sklearn.datasets
from matplotlib import pyplot as plt


X, y = sklearn.datasets.make_moons(200, noise=0.2)
# X, y = sklearn.datasets.make_blobs(n_features=2, centers=2)
ycopy = y
ytmp = [None]*len(y)
# for i in range(len(y)):
#     if y[i] == 0:
#         ytmp[i] = [1,0,0]
#     if y[i] == 1:
#         ytmp[i] = [0,1,0]
#     else:
#         ytmp[i] = [0,0,1]

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
nn_nlayer = 2

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

    dLdb[nn_nlayer] = (yhat-y)
    dLdW[nn_nlayer] = np.dot(z[nn_nlayer-1].T,dLdb[nn_nlayer])

    i = nn_nlayer
    while i > 0:
        i -= 1
        dLdb[i] = dLdb[i+1].dot(np.transpose(W[i+1])) * (1-z[i]*z[i])
        if i != 0:
            dLdW[i] = np.dot(z[i-1].T,dLdb[i])
        else:
            dLdW[i] = np.dot(X.T,dLdb[i])

    sum_dLdb = [np.sum(dLdb_element,axis=0) for dLdb_element in dLdb]
    return [dLdW,sum_dLdb]


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
        else:
            if i != nn_nlayer:
                a[i] = np.dot(z[i-1],W[i]) + b[i]
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
    eps = 0.001

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

        model = [W,b]

    return model

model = build_model(nn_hdim,num_passes=20000)

# X, y = sklearn.datasets.make_moons(200, noise=0.2)
y = ycopy
plotDecisionBoundary(model)
plt.savefig('decisionBoundary.png')
