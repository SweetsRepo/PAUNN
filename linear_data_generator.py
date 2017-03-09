import random

def linearDist(m,x,yint):
    return float(m*x+yint)

def variance(percent,value):
    randomval = random.random()*percent
    if(randomval<percent/2):
        return value - randomval*value
    else:
        return value + randomval*value

def genDistData(min, max, num, percent, m, yint):
    outputList = []
    for x in range(0, num):
        val = random.randint(min, max)
        y = variance(percent,linearDist(m,val,yint))
        outputList+=[(val,y)]
    with open('simulatedDist.csv', 'w') as linFile:
        for z in range(0,len(outputList)):
            linFile.write(str(outputList[z][0]) + ',' + str(outputList[z][1]) + '\n')

genDistData(5,60, 50, .1, 18.0/60.0, 2)
