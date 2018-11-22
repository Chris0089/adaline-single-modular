'''
Perceptron OOP

Activation functions available:
    step
    logistic

Training methods available:
    fast-forwarding (pending)
    backpropagation (pending)
'''
import sys
import random
import numpy as np

LAYER = 1
NEURON = 1
LIMIT_TOP = 1
LIMIT_BOTTOM = -1
ERROR101= "Error 101: No inputs found for a neuron."
ERROR102 = "Error 102: Too many input list for a neuron."
ERROR103 = "Error 103: Not especified activation function."
INPUTDATA = [[0,0],[0,1],[1,0],[1,1]]

class Neuron:
    inputData = None
    inputQuantity = None
    combinationOfInputs = None
    weightData = None
    output = None
    bias = None
    summation = None
    v = None
    activationFunction = None

    def __init__(self, inputs, af, bias):
        self.activationFunction = af
        self.bias = bias
        self.set_inputs(inputs)
        self.obtain_inputs_info()
        self.generate_random_values()
        self.calculate_output()
        
    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            self.inputData = []
            self.inputData = inputs.copy()
        else:
            self.inputData = inputs
    
    def obtain_inputs_info(self):
        if isinstance(self.inputData, list):
            if isinstance(self.inputData[0], list):
                if isinstance(self.inputData[0][0], list):
                    sys.exit(ERROR102)
                else:
                    self.inputQuantity = len(self.inputData[0])
                    self.combinationOfInputs = len(self.inputData)
            else:
                self.inputQuantity = len(self.inputData)
                self.combinationOfInputs = 1
        else:
            self.inputQuantity = 1
            self.combinationOfInputs = 1

    def generate_random_values(self):
        if self.inputQuantity > 1:
            self.weightData = []
            for eachInput in range(0,self.inputQuantity):
                self.weightData.append(random.uniform(LIMIT_BOTTOM, LIMIT_TOP))
        elif self.inputQuantity == 1:
            self.weightData = random.uniform(LIMIT_BOTTOM, LIMIT_TOP)
        else:
            sys.exit(ERROR101)
        self.bias = random.uniform(LIMIT_BOTTOM, LIMIT_TOP)
    
    def activation_function(self, value):
        if self.activationFunction == "step":
            if value <= 0:
                output = 0
            else:
                output = 1
            return output    
        elif self.activationFunction == "logistic":
            output = 1 / (1 + np.exp(value * -1))
            return output
        else:
            sys.exit(ERROR103)

    def calculate_output(self):
        if self.combinationOfInputs == 1:
            self.summation = 0
            self.v = 0
            if self.inputQuantity > 1:
                for quantity in range(0, self.inputQuantity):
                    self.summation += self.inputData[quantity] * self.weightData[quantity]
            else:
                self.summation = self.inputData * self.weightData
            self.v = self.summation + self.bias
            self.output = self.activation_function(self.v)
        else:
            self.summation = []
            self.v = []
            self.output = []
            for combination in range (0, self.combinationOfInputs):
                self.summation.append(0)
                for quantity in range(0, self.inputQuantity):
                    self.summation[combination] += self.inputData[combination][quantity] * self.weightData[quantity]
                self.v.append(0)
                self.v[combination] = self.summation[combination] + self.bias
                self.output.append(self.activation_function(self.v[combination]))
        print("Output = " + str(self.output))

class Layer:
    bias = None
    neurons = []
    neuronsQuantity = 1
    activationFunction = None
    training  = None

    def __init__(self, actFunc, trainingMtd):
        self.activationFunction = actFunc
        self.training = trainingMtd
        self.create_neurons() 

    def generate_rand_bias(self):
        self.bias = random.uniform(LIMIT_BOTTOM, LIMIT_TOP)

    def create_neurons(self):
        for neuron in range(0, self.neuronsQuantity):
            self.neurons.append(Neuron(INPUTDATA, self.activationFunction, self.bias ))

class NeuronalNetwork:
    layers = []
    desiredOutput = []
    finalOutput = []
    activationFuntion = "logistic"
    isDesiredOutput = False

    def __init__(self, desiredOutput):
       self.desiredOutput = desiredOutput.copy()

    def isDesiredOutput():
        if self.desiredOutput == self.finalOutput:
            return True
        else:
            return False


'''
pancho = Neuron(1, AF)
mencho = Neuron([2,3], AF)
mincho = Neuron([[2,3,4],[2,0,1]], AF)
'''
a = Layer("step", "fastforwarding")