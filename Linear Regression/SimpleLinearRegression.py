import numpy as np
from statistics import mean

class linearRegression:
    a=None
    b=None
        
    def fit(self,X, Y):
        self.a = ( (mean(X) * mean(Y)) - mean( X * Y ) ) / ( (mean(X))**2 - mean(X*X) )
        self.b = mean(Y) - self.a * mean(X)

    def predict(self,X):
        prediction = ((self.a*X)+self.b)
        return prediction


