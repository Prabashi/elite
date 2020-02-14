import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#initialize the weights and bias
def initializeWeightsAndBias(m):
    
    w = np.zeros((m,1))
    b = 0
    
    return w , b

def sigmoid(X):
    return 1/(1 + np.exp(- X))    

def propogate(X, Y, w, b):
    
    no_of_training_samples = X.shape[1]

    #Forward Propogation, calculating the cost
    Z = np.dot(w.T, X) + b;    
    A = sigmoid(Z)
    cost= -(1/no_of_training_samples) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    
    #Back Propogation , calculating the gradients
    dw = (1/no_of_training_samples)* np.dot(X, (A-Y).T)
    db = (1/no_of_training_samples)* np.sum(A-Y)
    
    grads= {"dw" : dw, "db" : db}
    
    return grads, cost

def gradientDescent(X, Y, w, b, num_of_iterations, alpha):
    
    costs=[] 
    
    for i in range(num_of_iterations):
 
        grads, cost = propogate(X, Y, w, b)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - alpha * dw
        b = b - alpha * db
        
        if i% 100 == 0:
            costs.append(cost)
             
    parameters = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}
    
    return parameters, grads, costs

def predict(X, w, b):
    
    no_of_samples = X.shape[1]
    
    y_prediction =  np.zeros((1,no_of_samples))
    
    w = w.reshape(X.shape[0], 1)
    
    A=sigmoid(np.dot(w.T, X)+b)
    
    
    for i in range(A.shape[1]):
        
        if(A[0,i]<0.5):
            y_prediction[0,i]=0
        else:
            y_prediction[0,i]=1
            
    return y_prediction

def logisticRegression(Xtrain, Ytrain, num_of_iterations, alpha):
    
    no_of_features = Xtrain.shape[0]
    
    w,b = initializeWeightsAndBias(no_of_features)
    
    parameters, grads, costs = gradientDescent(Xtrain, Ytrain, w, b, num_of_iterations, alpha) 
    
    w = parameters["w"]
    b = parameters["b"]
        
    
    d={"w":w, "b":b, "costs": costs}
    
    return d
    
df=pd.read_csv("breast-cancer-wisconsin.csv")

df.iloc[:, -1]=df.iloc[:, -1].map({2:1,4:0})
#df = df.dropna()
df = df.fillna(1)

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.349, random_state=42)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

data_model = logisticRegression(train_x.T, train_y.T, num_of_iterations=100000, alpha=0.001)

costs = data_model["costs"]
w = data_model["w"]
b = data_model["b"]

plt.plot(costs)
plt.title("Cost Vs Iterations")
plt.xlabel("No of Iterations")
plt.ylabel("Cost")
plt.show()

Y_prediction_train = predict(train_x.T, w, b)
Y_prediction_test = predict(test_x.T, w, b)

print("\nTrain Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y.T)) * 100))
print("\nTest Accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y.T)) * 100))
