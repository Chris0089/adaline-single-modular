'''
Perceptron OOP
'''

import random
import numpy as np
import pylab
import matplotlib.pyplot as plt

LIMIT_TOP = 1
LIMIT_BOTTOM = -1
ERROR101= "Error 101: No inputs found for a neuron."

class Neuron:
    inputData = None
    inputQuantity = None
    combinationOfInputs = None
    weightData = None
    output = None

    def get_inputs(self, singleInput):


    def generate_random_values(self):
        if self.inputQuantity > 0:
            self.weightData = []
            for eachInput in range(0,self.inputQuantity):
                self.weightData.append(random.uniform(LIMIT_BOTTOM, LIMIT_TOP))
        elif self.inputQuantity == 1:
            self.weightData = random.uniform(LIMIT_BOTTOM, LIMIT_TOP)
        else:
            print(ERROR101)