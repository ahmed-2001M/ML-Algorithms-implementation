import numpy as np



class logisticRegression:
    def __init__(self,learningRate = .001 , n_iteration = 1000) :
        self.alpha = learningRate
        self.iteration = n_iteration
        self.weights = None
        self.bias = None
    
    def fit(self , X , Y):
        samples , features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.iteration):
            linear = np.dot(X , self.weights) + self.bias
            prediction = self.sigmoid(linear)
            
            dw = (1/samples) * np.dot(X.T,(prediction - Y))
            db = (1/samples) * np.sum(prediction - Y)
            
            self.weights -= self.alpha*dw
            self.bias -= self.alpha*db
    
    def predict(self, X):
        linear = np.dot(X , self.weights) + self.bias
        prediction = self.sigmoid(linear)
        res =[1 if i>.5 else 0 for i in prediction]
        return res
    
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))