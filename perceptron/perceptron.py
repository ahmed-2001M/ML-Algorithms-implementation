import numpy as np

class Perceptron:
    def __init__(self,learning_rate=.01 , n_iters=1000 ):
        self.alph = learning_rate
        self.iterations = n_iters
        self.activation_function = self.unit_step_fun
        self.weights = None
        self.bias = None
    
    def fit(self,X,Y):
        
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        y = np.array([1 if i >0 else 0 for i in Y])
        
        for i in range(self.iterations) :
            for idx , x_i in enumerate(X):
                linear_output = np.sum(x_i * self.weights) + self.bias
                prediction = self.activation_function(linear_output)
                
                update = self.alph * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
    
    
    def predict(self, X) :
        linear_output = np.sum(X * self.weights) + self.bias
        prediction = self.activation_function(linear_output)
        return prediction

    def unit_step_fun(self,x):
        return 1 if x>0 else 0