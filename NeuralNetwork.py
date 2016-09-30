__author__ = "Christopher Sweet - crs4263@rit.edu"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"PAUNN - Predictive Analysis Using Neural Networks
"  Using a Neural Network and supervised learning the
"  following code will take user Input on biking data
"  and Output relevant predictive statistics.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib as mpl
from tkinter import *
from tkinter import ttk
import csv

class Menu():
    """
    Tkinter to get a basic basic UI
    """

    master = Tk()
    master.minsize(width=800, height=450)
    pages = ttk.Notebook(master)
    pages.winfo_width()

    openMenuFrame = Frame(pages)
    openMenuFrame.pack()

    #Contains the file name box and load button
    upperFrame = Frame(openMenuFrame)
    upperFrame.pack( side = TOP)
    loadfileButton = Button(upperFrame, text = 'Load', fg = 'black')
    loadfileButton.pack( side = RIGHT )
    loadfileLabel = Label(upperFrame, text = 'Load a *.csv File')
    loadfileLabel.pack( side = LEFT)
    filenameEntry = Entry(upperFrame)
    filenameEntry.pack(side = BOTTOM)

    #Contains the Run Button
    lowerFrame = Frame(openMenuFrame)
    lowerFrame.pack( side = BOTTOM)
    runButton = Button(lowerFrame, text = 'Run', fg = 'black')
    runButton.pack( side = BOTTOM)

    #Adds the first page in
    pages.add(openMenuFrame, text="Load")

    showDataFrame = Frame(pages)
    showDataFrame.pack()

    # Contains the Predict Button & Return Button
    lowerFrame2 = Frame(showDataFrame)
    lowerFrame2.pack(side=BOTTOM)
    returnButton = Button(lowerFrame2, text='Return', fg='black')
    returnButton.pack(side=LEFT)
    predictButton = Button(lowerFrame2, text='Predict', fg='black')
    predictButton.pack(side=RIGHT)

    #Adds the second page in
    pages.add(showDataFrame, text="Visualize")

    pages.pack()
    master.mainloop()

def nonlin(x,deriv = False):
    """
    Defines the sigmoid function used in refining matrix values
    :param x: Value to refine
    :param deriv: Flag to take Derivative
    :return: Value refined
    """
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def readData(fname):
    """
    Read in *.csv file that contains the appropriate biking data
    :param fname: File to read in
    :return: Input times, Output distances
    """
    with open(fname, newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter = ',', quotechar = '|')
        inputArray = np.array([[]])
        outputArray = np.array([[]])
        for row in dataReader:
            inputArray = np.append(inputArray, row[0])
            outputArray = np.append(outputArray, row[1])
    csvfile.close()
    return inputArray, outputArray

def predict(model, time):
    """
    Make a prediciton based off the learned values
    :param model: Trained Neural Network Parameters
    :param time: Time to produce a distance estimate for
    :return: Distance Estimate
    """
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

def scale_down(array0, array1, maxVals):
    """
    Scales the values down into unit measurements
    :param array0: First array to scale down
    :param array1: Second array to scale down
    :param maxVals: Maximum values to use for unit scaling
    :return: Scaled values
    """
    loop = np.nditer(array0, op_flags=['readwrite'], flags=['f_index'])
    while not loop.finished:
        array0[loop.index] = (array0[loop.index])/(maxVals[0])
        array1[loop.index] = (array1[loop.index])/(maxVals[1])
        loop.iternext()

def scale_up(array, prediction, maxVals):
    """
    Scales the values back up to distances
    :param array: Array of unit scalars
    :param prediction: Predicted value
    :param maxVals: Maximum values to use in scaling
    :return: Array Scaled back into Distance
    """
    loop = np.nditer(array, op_flags = ['readwrite'], flags = ['f_index'])
    while not loop.finished:
        array[loop.index] = prediction[loop.index]*maxVals[1]
        loop.iternext()

def create_model(fname):
    """
    Creates and trains a 3-layer Neural Network for cycling times/distances
    :param fname: File to load in
    :return: Dictionary of weighted Neural Network Parameters
    """
    # Reads the training data from csv
    inputData, outputData = readData(fname)
    # Input Data - Duration of Ride (min)
    inputData = inputData.T
    # Output Data - Distance traveled (Mi)
    outputData = outputData.T
    # Store the max values, and use them to normalize the data
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

menu = Menu()
#fname = input("Enter a csv filename: ")
#neuralNetwork = create_model(fname)
#distance = predict(neuralNetwork, 60)
#print("Based off your previous data you should be able to ride", distance, "miles!")