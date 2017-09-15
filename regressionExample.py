import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import time
 
# create Trainig Dataset
# X = np.linspace(-10,10,1000).reshape(-1,1)
# y = np.array(np.sin(X) +np.sin(X*2)+np.sin(X*5)).reshape(-1,1)
# y = np.ravel(y)
#create neural net regressor
reg = MLPRegressor(hidden_layer_sizes=(30,30),solver="lbfgs",max_iter=10000,alpha = 0.01)
# reg.fit(X,y)

# fig = plt.figure()
# ax = fig.gca()
# ax.plot(X,y,color='red')
# xtest = np.linspace(-10,10,2000).reshape(-1,1)
# plt.plot(xtest,reg.predict(xtest),color='black')
# fig.savefig('test.png')

# plt.plot(X,reg.predict(X))
#test prediction

 
# Create determinant testset

MatList = []
detList = []
N = 3
for i in range(10000):
    MatList.append(np.random.rand(N**2))
    detList.append(np.linalg.det(MatList[i].reshape(N,N)))
    
MatList = np.array(MatList)
detList = np.array(detList)
reg.fit(MatList,detList)









Ntest = 100
error = 0
timedet = 0
timepredict = 0

for i in range(Ntest):
    if np.mod(i,10)==0:
        print(i)
    
    testmat = np.random.rand(N**2)
    t1 = time.time()
    testdet = np.linalg.det(testmat.reshape(N,N))
    timedet += time.time() - t1

    t1 = time.time()
    predictdet = reg.predict([testmat])
    timepredict += time.time() - t1
    
    error += (testdet-predictdet)**2

print('Error = %g' %(error/Ntest))
print(timedet)
print(timepredict)
