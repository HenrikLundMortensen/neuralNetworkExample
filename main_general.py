import numpy as np
import sklearn.datasets
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import matplotlib.animation as manimation



FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

X, y = sklearn.datasets.make_moons(200, noise=0.2)
# X, y = sklearn.datasets.make_blobs(n_features=2, centers=3,n_samples=1000)
# X, y = sklearn.datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
#                              n_clusters_per_class=1)

ycopy = y.copy()
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

def calculateLoss(yhat):
    """
    """

    correct_logprobs = np.sum(-y*np.log(yhat),axis=1)
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
        dLdb[i] = dLdb[i+1].dot(W[i+1].T) * (1-np.power(z[i],2))
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
            a[i] = x.dot(W[i])+ b[i]
            z[i] = np.tanh(a[i])
        else:
            if i != nn_nlayer:
                a[i] = z[i-1].dot(W[i]) + b[i]
                z[i] = np.tanh(a[i])
            else:
                a[i] = z[i-1].dot(W[i]) + b[i]

    return [softmax(a[nn_nlayer]),z]

def plotDecisionBoundary(model,fig):
    """
    """
    
    # Create mesh grid

    xmin = min(X[:,0])-0.5
    xmax = max(X[:,0])+0.5
    ymin = min(X[:,1])-0.5
    ymax = max(X[:,1])+0.5

    # xmin = -20
    # xmax = 20
    # ymin = -20
    # ymax = 20

    h = 0.05


    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))

    Z = np.argmax(predict(model,np.c_[xx.ravel(),yy.ravel()])[0],axis=1)
    Z = Z.reshape(xx.shape)
    ax = fig.gca()
    ax.contourf(xx,yy,Z,cmap = plt.cm.Spectral)
    ax.scatter(X[:,0], X[:,1], s=40, c=ycopy, cmap=plt.cm.Spectral)
    ax.set_xticks([])    
    ax.set_yticks([])    


def build_model(nn_hdim,num_passes=1):
    """
    """
    eps = 0.001
    reg_lambda = 0.0000

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
    fig = plt.figure()
    with writer.saving(fig, "writer_test.mp4",dpi=200):
        for i in range(num_passes):
            [yhat,z] = predict(model,X)

            [dLdW,dLdb] = getGradient(model,X,yhat,y,z)

            for j in range(nn_nlayer+1):
                W[j] += -eps*dLdW[j]
                W[j] += reg_lambda*W[j]            

                b[j] += -eps*dLdb[j]
            model = [W,b]    

            if np.mod(i,50) == 0:
                loss = calculateLoss(yhat)

                print(loss)
                plotDecisionBoundary(model,fig)
                writer.grab_frame()
                if loss < 0.05:
                    break
        
    return model

model = build_model(nn_hdim,num_passes=10000)
# X, y = sklearn.datasets.make_moons(200, noise=0.2)
# y = ycopy

