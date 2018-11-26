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
    def __init__(self, inputs, af, bias):
        self.inputData = None
        self.inputQuantity = None
        self.combinationOfInputs = None
        self.weightData = None
        self.output = None
        self.summation = None
        self.v = None
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
    def __init__(self, inputData, desiredData, actFunc, trainingMtd, neurons):
        self.bias = None
        self.neurons = []
        self.neuronsQty = None
        self.inputData = inputData.copy()
        self.outputData = None
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
    
    def training_logistic(self):
        for neuron in range(0, self.neuronsQty):
            for out in range(0, len(self.desiredOutput)):
                error = 0.0
                error = self.desiredOutput[out] - self.outputData[out]
                if abs(error) > MAXERROR:
                    self.bias = self.bias  + (ETA * error * self.outputData[out] * (1-self.outputData[out]))
                    self.neurons[neuron].bias = self.bias
                    for inputValue in range(0, len(self.inputData[0])):
                        self.neurons[neuron].weightData[inputValue] = self.neurons[neuron].weightData[inputValue] + (ETA * error * self.inputData[out][inputValue]  * self.outputData[out] * (1-self.outputData[out]))

class NeuronalNetwork:
    def __init__(self, inputData, desiredData, eta, layersQty, neuronsQty, typeNetwork="other"):
        self.finalOutput = []
        self.finalWeights = []
        self.iteration = 0
        self.inputData = inputData.copy()
        self.desiredOutput = []
        self.desiredOutput = desiredData.copy()
        self.eta = eta
        self.layers = []
        self.layersQty = layersQty
        self.neuronsQty = neuronsQty
        self.typeNetwork = typeNetwork
        self.mainAlgorithm()

    def isDesiredOutput(self):
        if self.typeNetwork == "perceptron":
            if self.desiredOutput == self.finalOutput:
                return True
            else:
                return False
        elif self.typeNetwork == "adaline":
            for val in range(0, len(self.desiredOutput)):
                error = abs(self.desiredOutput[val] - self.finalOutput[val])
                if error > MAXERROR:
                    return False
            return True

    
    def mainAlgorithm(self):
        if self.typeNetwork == "perceptron":
            iteration = 0
            layer = Layer(self.inputData, self.desiredOutput, "step", "fastforwarding", 1)
            while not self.isDesiredOutput():
                iteration += 1
                print( "Ciclo: " + str(iteration) )
                layer.training_step()
                layer.calculate_output()
                self.finalOutput = layer.outputData.copy()
                print(self.finalOutput)
                print(self.desiredOutput[0])
                self.iteration = iteration
            print("finished!!!!!!!!")   
            self.printPlot(layer.neurons[0].weightData[0], layer.neurons[0].weightData[1], layer.neurons[0].bias)
        if self.typeNetwork == "adaline":
            iteration = 0
            layer = Layer(self.inputData, self.desiredOutput, "logistic", "fastforwarding", 1)
            while not self.isDesiredOutput():
                iteration += 1
                print( "Ciclo: " + str(iteration) )
                layer.training_adaline()
                layer.calculate_output()
                self.finalOutput = layer.outputData.copy()
                print(self.finalOutput)
                print(self.desiredOutput[0])
                self.iteration = iteration
            print("finished!!!!!!!!")   
            self.printPlot(layer.neurons[0].weightData[0], layer.neurons[0].weightData[1], layer.neurons[0].bias)

    def printPlot(self, w1, w2, bias):
        minX = min(chain.from_iterable(self.inputData)) - 2
        maxX = max(chain.from_iterable(self.inputData)) + 2
        x = np.linspace(minX, maxX, 4)
        formulaPlot = (-1 * bias / w2) - (w1 / w2 * x)
        if self.desiredOutput[0] == 0:
            plt.plot(0,0, 'x', color = 'red')
        else:
            plt.plot(0,0, 'ro', color = 'green')
        if self.desiredOutput[1] == 0:
            plt.plot(0,1, 'x', color = 'red')
        else:
            plt.plot(0,1, 'ro', color = 'green')
        if self.desiredOutput[2] == 0:
            plt.plot(1,0, 'x', color = 'red')
        else:
            plt.plot(1,0, 'ro', color = 'green')
        if self.desiredOutput[3] == 0 :
            plt.plot(1,1, 'x', color = 'red')
        else:
            plt.plot(1,1, 'ro', color = 'green')
        plt.suptitle(str(self.typeNetwork) + ". " + str(self.iteration) + " iteraciones.")
        pylab.plot(x, formulaPlot, color = "blue")
        pylab.show()
                
perceptron = NeuronalNetwork(INPUTDATA, DESIRED_VALUES, ETA, LAYERS, NEURONS, "adaline" )
adaline = NeuronalNetwork(INPUTDATA, DESIRED_VALUES, ETA, LAYERS, NEURONS, "adaline" )

