from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y),10))
    for i in range(len(y)):
        y_vect[i,y[i]] = 1
    return y_vect

def f(x):
    return 1/(1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1-f(x))


digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
nn_structure = [64, 30, 10]
