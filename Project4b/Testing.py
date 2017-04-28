from NeuralNetUtil import buildExamplesFromCarData, buildExamplesFromPenData, buildExamplesFromExtraData, buildExamplesFromXorData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [10]):
    return buildNeuralNet(carData, maxItr = 200, hiddenLayerList = hiddenLayers)

xorData = buildExamplesFromXorData()
def testXorData(hiddenLayers = [0]):
    return buildNeuralNet(xorData, maxItr = 300, hiddenLayerList = hiddenLayers)

extraData = buildExamplesFromExtraData()
def testExtraData(hiddenLayers = [15]):
    return buildNeuralNet(extraData, maxItr = 200, hiddenLayerList = hiddenLayers)

###################################################################################

def getDatasetStat(func = testPenData, numIter = 5, numPercep = None):
	temp = []
	result = []

	if numPercep == None:
		for i in range(numIter):
			nnet, testAccuracy = func()
			temp.append(testAccuracy)
	else:
		for i in range(numIter):
			nnet, testAccuracy = func(numPercep)
			temp.append(testAccuracy)
	result.append([average(temp), max(temp), stDeviation(temp)])
	return result

if __name__=='__main__':
	"""
		README:
		
		All helpful functions are well set. If you want to test specific question,
		just uncomment the related lines below.
		
		To save your time, you'd better run each question once at a time.
	"""

	# """ Question 5: Learning With Restarts """
	# print "Pen dataset result: \n", getDatasetStat(func = testPenData, numIter = 5)
	# print "Car dataset result: \n", getDatasetStat(func = testCarData, numIter = 5)

	# """ Question 6: Varying The Hidden Layer """
	# car = []
	# pen = []
	# for i in range(0, 45, 5):
	# 	car.append(getDatasetStat(func = testCarData, numIter = 5, numPercep = [i]))
	# 	pen.append(getDatasetStat(func = testPenData, numIter = 5, numPercep = [i]))
	# print car
	# print pen

	# """ Question 7: Learning XOR """
	xor = []
	for i in range(0, 30, 1):
		xor.append(getDatasetStat(func = testXorData, numIter = 5, numPercep = [i]))
	print xor

	# """ Question 8: Novel Dataset """
	# print "Extra dataset result: \n", getDatasetStat(func = testExtraData, numIter = 5)
