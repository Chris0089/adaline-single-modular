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
import pylab
import matplotlib.pyplot as plt
from itertools import chain
from user_data import *


LIMIT_TOP = 1
LIMIT_BOTTOM = -1
ERROR101= "Error 101: No inputs found for a neuron."
ERROR102 = "Error 102: Too many input list for a neuron."
ERROR103 = "Error 103: Not especified activation function."


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
        print("Weight = " + str(self.weightData))
        print("Bias = " + str(self.bias))

class Layer:
    bias = None
    neurons = []
    neuronsQty = None
    activationFunction = None
    training  = None
    inputData = None
    outputData = None
    desiredOutput = None

    def __init__(self, inputData, desiredData, actFunc, trainingMtd, neurons):
        self.inputData = inputData.copy()
        self.desiredOutput = desiredData.copy()
        self.activationFunction = actFunc
        self.training = trainingMtd
        self.neuronsQty = neurons
        self.generate_rand_bias()
        self.create_neurons() 

    def generate_rand_bias(self):
        self.bias = random.uniform(LIMIT_BOTTOM, LIMIT_TOP)

    def create_neurons(self):
        for neuron in range(0, self.neuronsQty):
            self.neurons.append(Neuron(INPUTDATA, self.activationFunction, self.bias ))
            self.outputData = self.neurons[neuron].output.copy()
    
    def calculate_output(self):
        if self.neuronsQty == 1:
            self.neurons[0].calculate_output()
            self.outputData = self.neurons[0].output.copy()
    
    def training_step(self):
        for neuron in range(0, self.neuronsQty):
            for out in range(0, len(self.desiredOutput)):
                error = 0.0
                error = self.desiredOutput[out] - self.outputData[out]
                if error != 0:
                    self.bias = self.bias  + (ETA * error)
                    self.neurons[neuron].bias = self.bias
                    for inputValue in range(0, len(self.inputData[0])):
                        self.neurons[neuron].weightData[inputValue] = self.neurons[neuron].weightData[inputValue] + (ETA * error * self.inputData[out][inputValue])

class NeuronalNetwork:
    inputData = []
    desiredOutput = []
    eta = None
    layers = []
    layersQty = None
    neuronsQty = None
    finalOutput = []
    finalWeights = []
    typeNetwork = None

    def __init__(self, inputData, desiredData, eta, layersQty, neuronsQty, typeNetwork="other"):
        self.inputData = inputData.copy()
        self.desiredOutput = desiredData.copy()
        self.eta = eta
        self.layersQty = layersQty
        self.neuronsQty = neuronsQty
        self.typeNetwork = typeNetwork
        self.mainAlgorithm()

    def isDesiredOutput(self):
        if self.desiredOutput == self.finalOutput:
            return True
        else:
            return False
    
    def mainAlgorithm(self):
        if self.typeNetwork == "perceptron":
            for col in range(0, len(self.desiredOutput)):
                iteration = 0
                self.finalOutput.append([])
                layer = Layer(self.inputData, self.desiredOutput[col], "step", "fastforwarding", 1)
                while not self.isDesiredOutput():
                    iteration += 1
                    print( "Ciclo: " + str(iteration) )
                    layer.training_step()
                    layer.calculate_output()
                    self.finalOutput[col] = layer.outputData.copy()
                    print(self.finalOutput)
                    print(self.desiredOutput[0])
                print("finished!!!!!!!!")
                self.printPlot(layer.neurons[0].weightData[0], layer.neurons[0].weightData[1], layer.neurons[0].bias)

    def printPlot(self, w1, w2, bias):
        for column in range(0, len(self.finalOutput)):
            #x2 = (self.bias[column] / self.weightData[column][1] - self.weightData[column][0]/ self.weightData[column][1] * self.inputData[0][0])
            minX = min(chain.from_iterable(self.inputData)) - 2
            maxX = max(chain.from_iterable(self.inputData)) + 2
            x = np.linspace(minX, maxX, 4)
            formulaPlot = (-1 * bias / w2) - (w1 / w2 * x)
            if self.desiredOutput[column][0] == 0:
                plt.plot(0,0, 'x', color = 'red')
            else:
                plt.plot(0,0, 'ro', color = 'green')
            if self.desiredOutput[column][1] == 0:
                plt.plot(0,1, 'x', color = 'red')
            else:
                plt.plot(0,1, 'ro', color = 'green')
            if self.desiredOutput[column][2] == 0:
                plt.plot(1,0, 'x', color = 'red')
            else:
                plt.plot(1,0, 'ro', color = 'green')
            if self.desiredOutput[column][3] == 0 :
                plt.plot(1,1, 'x', color = 'red')
            else:
                plt.plot(1,1, 'ro', color = 'green')
            pylab.plot(x, formulaPlot, color = "blue")
            pylab.show()
                

'''
pancho = Neuron(1, AF)
mencho = Neuron([2,3], AF)
mincho = Neuron([[2,3,4],[2,0,1]], AF)
'''
perceptron = NeuronalNetwork(INPUTDATA, DESIRED_VALUES, ETA, LAYERS, NEURONS, "perceptron" )
