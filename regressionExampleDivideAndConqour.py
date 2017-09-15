import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor


def getGuessCost(m):
    """

    """
    
    return guess_cost_cst*(m**guess_cost_potens)

def T(m):
    """

    """
    return cost_cst*(m**cost_potens)

def getCost(m,n):
    """
    m: Size of original matrix
    n: Size of matrix we can guess

    """
    s = np.log2(m)

    cost = 0
    for i in range(int(s)+1):
        r = 2**i
        print('m=%g,n=%g,cost=%g \t r = %g' %(m,n,cost,r))
        if n==m/r:
            if n!=1:
                cost += r*getGuessCost(n)
            break
        else:
            cost += r*T(m/r)
        print('m=%g,n=%g,cost=%g \t r = %g' %(m,n,cost,r))            
    return cost
for hls in [1,10,20,30,40,50]:
    
    
    reg = MLPRegressor(hidden_layer_sizes=(hls,hls),solver="lbfgs",max_iter=10000,alpha = 0.01)
    Ntrain = 10000
    Ntest = 100
    n = 2**2
    m = 2**6

    # Get training data
    X = [None]*Ntrain
    y = [None]*Ntrain

    for i in range(Ntrain):
        a = np.zeros((n,n))
        diag = np.random.rand(n)
        offdiag = np.random.rand(n-1)
        for j in range(n):
            a[j,j] = diag[j]

        for j in range(n-1):
            a[j,j+1] = offdiag[j]
            a[j+1,j] = offdiag[j]        

        X[i] = np.array(a).ravel()
        y[i] = np.linalg.eigvals(a)

    X = np.array(X)
    y = np.array(y)


    # Get test data
    Xtest = [None]*Ntest
    ytest = [None]*Ntest

    for i in range(Ntest):
        a = np.zeros((n,n))
        diag = np.random.rand(n)
        offdiag = np.random.rand(n-1)
        for j in range(n):
            a[j,j] = diag[j]

        for j in range(n-1):
            a[j,j+1] = offdiag[j]
            a[j+1,j] = offdiag[j]

        Xtest[i] = np.array(a).ravel()
        ytest[i] = np.linalg.eigvals(a)

    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    reg.fit(X,y)
    error = sum(sum(np.power(ytest-reg.predict(Xtest),2)))/4/Ntest
    print(error)


      








