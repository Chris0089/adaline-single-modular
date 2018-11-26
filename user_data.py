'''
Perceptron OOP
    
AF -> Activation function. These are the activation function availables:
    step
    logistic

Training methods available:
    fast-forwarding (pending)
    backpropagation (pending)
'''

INPUTDATA = [[0,0],[0,1],[1,0],[1,1]]
DESIRED_VALUES = [0,0,0,1]
LAYERS = 1
NEURONS = 1
ETA = 0.2 
AF = "step"
TM = "fast-forward"
MAXERROR = 0.1