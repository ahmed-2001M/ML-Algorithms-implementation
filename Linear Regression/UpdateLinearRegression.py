import numpy as np 



class linearRegression:
    def __init__(self , learningRate = 0.001, n_iteration = 1000) :
        self.alpha = learningRate
        self.iterations = n_iteration
        self.weights = None
        self.bias = None
        
    def fit(self , X , Y):
        samples, features = X.shape
        
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.iterations):
            prediction = np.dot(X, self.weights) + self.bias
            
            dw = (1/samples) * np.dot(X.T,(prediction - Y))
            db = (1/samples) * np.sum(prediction - Y)
            
            self.weights -= self.alpha*dw
            self.bias -= self.alpha*db
    
    
    
    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        return prediction
