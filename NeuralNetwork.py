__author__ = "Christopher Sweet - crs4263@rit.edu"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"PAUNN - Predictive Analysis Using Neural Networks
"  Using a Neural Network and supervised learning the
"  following code will take user Input on biking data
"  and Output relevant predictive statistics.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np

# Sigmoid function which determines how much to refine values by.
def nonlin(x,deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Make a prediction based off the learned values
def predict(model, time):
    maxVals, syn0, syn1 = model['maxVals'], model['syn0'], model['syn1']
    # Scale down into unit scalar
    time /= maxVals[0]
    matTime = np.array([[time]])
    # Matrix Expansion
    matTime = nonlin(matTime.dot(syn0))
    # Matrix Contraction
    matTime = nonlin(matTime.dot(syn1))
    # Scale back up into distance
    matTime *= maxVals[1]
    # Return the value predicted
    return matTime[0][0][0]

# Scale the values down to Unit Scalars - Helper Function
def scale_down(array0, array1, maxVals):
    loop = np.nditer(array0, op_flags=['readwrite'], flags=['f_index'])
    while not loop.finished:
        array0[loop.index] = (array0[loop.index])/(maxVals[0])
        array1[loop.index] = (array1[loop.index])/(maxVals[1])
        loop.iternext()

# Scale the values back up to distances - Helper Function
def scale_up(array, prediction, maxVals):
    loop = np.nditer(array, op_flags = ['readwrite'], flags = ['f_index'])
    while not loop.finished:
        array[loop.index] = prediction[loop.index]*maxVals[1]
        loop.iternext()

def create_model():
    # Input Data - Duration of Ride (min)
    inputData = np.array([[63.0, 27.0, 53.0, 34.0, 48.0, 120.0, 66.0, 75.0, 125.0]]).T

    # Output Data - Distance traveled (Mi)
    outputData = np.array([[10.8, 4.28, 8.86, 8.28, 9.11, 21.6, 13.33, 12.55, 21.35]]).T

    # Store the max values, and use them to scale the data
    maxVals = np.array([[np.amax(inputData)], [np.amax(outputData)]])

    # Scale input times and output distances
    scale_down(inputData, outputData, maxVals)

    # Seed Random values to start with
    np.random.seed(1)

    # Create the synaptic connections between neurons.
    synapse0 = 2 * np.random.random((1, 10)) - 1
    synapse1 = 2 * np.random.random((10, 1)) - 1

    # Training the neural network. Number of runs influences results only to a certain point.
    for t in range(0, 50000):
        # Feed data forward from input to output
        layer0 = inputData
        layer1 = nonlin(layer0.dot(synapse0))
        layer2 = nonlin(layer1.dot(synapse1))

        # Determine how far off predicted values were from the training ones
        layer2_error = outputData - layer2

        # Calculate the amount to modify layer 2 weights by
        layer2_delta = layer2_error * nonlin(layer2, True)

        # Calculates the amount of error resultant from layer 1 (Look into how this actually accomplishes this)
        layer1_error = layer2_delta.dot(synapse1.T)

        # Calculate the amount to change layer 1 weights by
        layer1_delta = layer1_error * nonlin(layer1, True)

        # Update synaptic weights
        synapse1 += layer1.T.dot(layer2_delta)
        synapse0 += layer0.T.dot(layer1_delta)

    print("Done Training!")
    # Return an anonymous dictionary of the weighted network values
    return { 'maxVals': maxVals, 'syn1': synapse1, 'syn0': synapse0}

neuralNetwork = create_model()
distance = predict(neuralNetwork, 60)
print("Based off your previous data you should be able to ride", distance, "miles!")