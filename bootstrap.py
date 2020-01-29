import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

#%matplotlib inline

data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]
#X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)
#print(np.shape(Y_train), np.shape(X_train))
print(X.shape, Y.shape)

beta_0 = []
beta_1 = []
beta_2 = []
beta_3 = []
beta_4 = []
regresion = sklearn.linear_model.LinearRegression()
for i in range(1000):
    a = np.random.randint(0 , 69, size = 69)
    regresion.fit(X[a,:], Y[a])
    beta_0.append(regresion.intercept_)
    beta_1.append(regresion.coef_[0])
    beta_2.append(regresion.coef_[0])
    beta_3.append(regresion.coef_[0])
    beta_4.append(regresion.coef_[0])

plt.subplot(2,2,1)

plt.hist(beta_1)
plt.title(r'$\beta_{1} =$' + str(np.mean(beta_1))+ '$\pm$' + str(np.std(beta_1)))

plt.subplot(2,2,2)
plt.hist(beta_2)
plt.title(r'$\beta_{2} =$' + str(np.mean(beta_2))+ '$\pm$' + str(np.std(beta_2)))


plt.subplot(2,2,3)
plt.hist(beta_3)
plt.title(r'$\beta_{3} =$' + str(np.mean(beta_3))+ '$\pm$' + str(np.std(beta_3)))

plt.subplot(2,2,4)
plt.hist(beta_4)
plt.title(r'$\beta_{4} =$' + str(np.mean(beta_4))+ '$\pm$' + str(np.std(beta_4)))

plt.savefig('bootstrap.png')
