import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import time

start_time = time.time()

data_matrix = np.array(pd.read_csv('SUSY.csv'))

X = data_matrix[:,1:]
Y = data_matrix[:,0]
Y[Y==0.0] = -1.0

nrows, ncols = X.shape[0], X.shape[1]

training_percent = 70

print "\n\n Creating training set ...\n"
X_train = X[:int(training_percent*nrows/100),:]
Y_train = Y[:int(training_percent*nrows/100)]
print "\n Training set created ...\n"
nrows_train, ncols_train = X_train.shape[0], X_train.shape[1]

print "\n\n Creating test set ...\n"
X_test = X[int(training_percent*nrows/100):,:]
Y_test = Y[int(training_percent*nrows/100):]
print "\n Test set created ...\n"
nrows_test, ncols_test = X_test.shape[0], X_test.shape[1]

w = np.zeros(ncols_train)

regularizer = 0.00001
Y_pred = np.dot(X_train, w)
error = float(np.logical_xor((Y_pred > 0).astype(int), (Y_train.astype(int) > 0).astype(int)).sum())/float(nrows_train)
count = 1
total_iterations = 1000
error_list = [error]
count_list = [count]

for t in range(total_iterations):
    
    i = np.random.randint(0, nrows_train, 1)[0]
    
    decision = np.dot(X_train[i,:], w)*Y_train[i]
    
    if decision >= 1:
        w_new = np.add(w, 0)
    
    elif decision < 1:
        w_new = np.add(w, np.divide(np.multiply(X_train[i],Y_train[i]), np.sqrt(count)))
        w_new = np.multiply(w_new, min(1, 1/(np.linalg.norm(w_new)*np.sqrt(regularizer))))
    
    w = w_new
    count = count + 1
    
    Y_pred = np.dot(X_test, w)
    
    error = float(np.logical_xor((Y_pred > 0).astype(int), (Y_test.astype(int) > 0).astype(int)).sum())/float(nrows_test)
    
    error_list.append(error)
    count_list.append(count)
    
plt.title('Variation of accuracy in prediction with increasing iterations \n')
plt.xlabel('Iteration number')
plt.ylabel('Test error')
plt.plot(count_list, error_list, color = '#7fffd4')
tck1 = interpolate.splrep(count_list, error_list, k = 3, s = 900)
error_list_int = interpolate.splev(count_list, tck1, der = 0)
plt.plot(count_list, error_list_int, color = 'magenta', label = 'Stochastic Gradient Descent')
plt.legend()         
plt.show()

end_time = time.time() - start_time
print "\n\nPercentage accuracy = " + str(100 - error_list[-1] * 100) + "%\n"
print "\nTime taken = " + str(end_time) + " seconds\n"