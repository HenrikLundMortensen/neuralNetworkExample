import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor as Regressor

X_train = np.linspace(0,10,100).reshape(-1,1)
y_train = np.array(np.sin(X_train)).reshape(-1,1)




nn = Regressor(hidden_layer_sizes=(200,200,), max_iter=10000,activation='relu',solver='lbfgs',alpha=0.001)
nn.fit(X_train, y_train)


plt.plot(X_train,nn.predict(X_train))
plt.plot(X_train,y_train)
plt.show()



