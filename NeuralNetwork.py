__author__ = "Christopher Sweet - crs4263@rit.edu"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"PAUNN - Predictive Analysis Using Neural Networks
"  Using a Neural Network and supervised learning the
"  following code will take user Input on biking data
"  and Output relevant predictive statistics.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import sys

class Menu():
    """
    Create the menu system for PAUNN
    Set all UI Elements in init function.
    Bind all buttons to callback functions.
    """
    def __init__(self):
        """
        Create the UI and set all attributes for usability
        """
        self.master = tk.Tk()
        self.master.minsize(width=800, height=450)
        self.master.maxsize(width=800, height=450)
        self.master.title("PAUNN")
        # Notebook settings
        self.pages = ttk.Notebook(self.master, width=750, height=400, padding=10)
        self.openMenuFrame = tk.Frame(self.pages)
        # Upper panel on the first page
        self.upperFrame = tk.Frame(self.openMenuFrame)
        self.loadfileButton = tk.Button(self.upperFrame, text='Select File', fg='black',
                                command = self.loadFileCallback)
        self.loadfileLabel = tk.Label(self.upperFrame, text='Load Biking Data: ')
        # Lower panel on the first page
        self.lowerFrame = tk.Frame(self.openMenuFrame)
        # Adds the first page in
        self.pages.add(self.openMenuFrame, text="Load")

        # 2nd Page of the notebook
        self.showDataFrame = tk.Frame(self.pages)
        # Upper panel on the second page. Broken into left and right
        self.upperFrame2 = tk.Frame(self.showDataFrame)
        self.upperFrame2Left = tk.Frame(self.upperFrame2)
        self.upperFrame2Right = tk.Frame(self.upperFrame2)
        self.timeLabel = tk.Label(self.upperFrame2Left, text = 'Time(min)', fg = 'black')
        self.timeEntry = tk.Entry(self.upperFrame2Left, width = 5, textvariable = tk.StringVar)
        self.distanceLabel = tk.Label(self.upperFrame2Left, text='Distance(mi)', fg='black')
        self.distEntry = tk.StringVar()
        self.distanceEntry = tk.Entry(self.upperFrame2Left, width = 5, state = tk.DISABLED,
                                      textvariable = self.distEntry, fg = 'black')
        self.predictButton = tk.Button(self.upperFrame2Left, text = 'Predict', fg ='black',
                               command = self.predictCallback)
        self.spacingLabel = tk.Label(self.upperFrame2Left, text = "   ")
        # Lower panel on the second page
        self.lowerFrame2 = tk.Frame(self.showDataFrame)
        self.resetButton = tk.Button(self.lowerFrame2, text = 'Reset', fg = 'black',
                                     bd = 2, command = self.resetCallback)

        # Adds the second page in
        self.pages.add(self.showDataFrame, text = "Visualize")

        # Setting the location of all elements using pack
        self.pages.pack()
        #Page 1
        self.upperFrame.config(pady = 20)
        self.lowerFrame.config(pady = 20)
        self.upperFrame.pack(side = tk.TOP)
        self.loadfileButton.pack(side = tk.RIGHT)
        self.loadfileLabel.pack(side = tk.LEFT)
        self.lowerFrame.pack(side = tk.BOTTOM)

        #Page 2
        self.upperFrame2.config(pady = 20)
        self.upperFrame2Left.config(width = 20)
        self.upperFrame2Right.config(width = 580)
        self.lowerFrame2.config(pady = 20)
        self.upperFrame2.pack(side = tk.TOP)
        self.upperFrame2Left.pack(side = tk.LEFT)
        self.upperFrame2Right.pack(side = tk.RIGHT)
        self.timeLabel.pack(side = tk.TOP)
        self.timeEntry.pack(side = tk.TOP)
        self.distanceLabel.pack(side = tk.TOP)
        self.distanceEntry.pack(side = tk.TOP)
        self.spacingLabel.pack(side = tk.TOP)
        self.predictButton.pack(side = tk.TOP)
        self.lowerFrame2.pack(side = tk.BOTTOM)
        self.resetButton.pack(side = tk.LEFT)

    def loadFileCallback(self):
        fname = askopenfilename()
        self.nn = create_model(fname)
        self.createPlot(fname)

    def predictCallback(self):
        try:
            self.distEntry.set(str(predict(self.nn, self.timeEntry.get())))
        except TypeError:
            print("Neural Network must be generated prior to prediction ")

    def resetCallback(self):
        self.master.destroy()
        self.mainloop()

    def createPlot(self, fname):
        x, y = readData(fname)
        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)
        a.scatter(x, y, s=10)
        canvas = FigureCanvasTkAgg(f, master=self.upperFrame2Right)
        canvas.show()
        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = 1)


    def mainloop(self):
        tk.mainloop()

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
    singlearray = np.genfromtxt(fname, dtype=float, delimiter=',')
    inputArray = singlearray[:,0].reshape(1, len(singlearray))
    outputArray = singlearray[:,1].reshape(1, len(singlearray))
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
    time = float(time) / maxVals[0]
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
    synapse0 = 2 * np.random.random((1, 5)) - 1
    synapse1 = 2 * np.random.random((5, 1)) - 1
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
menu.mainloop()
#fname = input("Enter a csv filename: ")
#neuralNetwork = create_model(fname)
#distance = predict(neuralNetwork, 30)
#print("Based off your previous data you should be able to ride", distance, "miles!")