import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        # print value
        return 1.0 / (1 + exp(-value))
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            float
            The value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        # Remember to add 1 to the start of inActs for the bias input.
        inActs = [1.0] + inActs
        # print len(inActs)
        output = self.sigmoid(self.getWeightedSum(inActs))
        return output
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        return self.sigmoid(value) * (1 - self.sigmoid(value))

    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        inActs = [1.0] + inActs
        output = self.sigmoidDeriv(self.getWeightedSum(inActs))
        return output

    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        """YOUR CODE"""
        inActs = [1.0] + inActs
        
        for i in range(len(self.weights)):
            weightChange = alpha * delta * inActs[i]
            self.weights[i] += weightChange
            totalModification += abs(weightChange)
        return totalModification
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer
            [16, 24, 10]: 16 inputs, 24 perceptron in first hidden layer, 10 outputs
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1 # numer of hidden layers + number of output layers
        
        #build hidden layer(s): one hidden layer contains 24 perceptrons        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer: output layer contains 10 perceptrons
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        """YOUR CODE"""
        # print self.layers[1][1] # two layers: one hidden layer(24 x 1), one output layer(10 x 1)
        # print len(inActs) # one example: 16 x 1
        outputs = [inActs]
        for layer in self.layers:
            outActs = []
            for perceptron in layer:
                outActs.append(perceptron.sigmoidActivation(inActs))
            outputs.append(outActs)
            inActs = outActs
        return outputs # [[input inActs list 16 x 1], [output outActs list 24 x 1], [output final list 10 x 1]]
    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        
        Args: 
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons, 
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """

        # print len(examples) # 2 x 1
        # keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0

        for example in examples:#for each example 16, 10
            #keep track of deltas to use in weight change
            # print example[1] #  output vector
            # print example[0] # input vector

            deltas = [] # num of examples x num of output: 2 x 10
            #Neural net output list
            allLayerOutput = self.feedForward(example[0])
            # print len(allLayerOutput[-1])
            # print len(allLayerOutput[-2])
            lastLayerOutput = allLayerOutput[-1]
            #Empty output layer delta list
            outDelta = [] # 10 x 1
            #iterate through all output layer neurons: 10 x 1
            for outputNum in xrange(len(example[1])):
                gPrime = self.outputLayer[outputNum].sigmoidActivationDeriv(allLayerOutput[-2])
                error = example[1][outputNum] - lastLayerOutput[outputNum] # expected - actual
                delta = gPrime * error
                averageError+=error*error/2
                outDelta.append(delta)
            deltas.append(outDelta)
            
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            """
            # print self.numHiddenLayers # 1
            for layerNum in xrange(self.numHiddenLayers-1,-1,-1): # from the last hidden layer till the first layer
                layer = self.layers[layerNum] # hidden layer + output layer
                nextLayer = self.layers[layerNum+1]
                hiddenDelta = []
                #Iterate through all neurons in this layer
                for neuronNum in xrange(len(layer)):
                    # layerNum = 0 which is the input layer, so basically, sigmoidActivationDeriv(previous layer output)
                    gPrime = layer[neuronNum].sigmoidActivationDeriv(allLayerOutput[layerNum])
                    weightedSum = 0.0
                    for neuronNumNext in xrange(len(nextLayer)):
                        weightedSum += nextLayer[neuronNumNext].weights[neuronNum + 1] * deltas[layerNum][neuronNumNext]
                    delta = gPrime * weightedSum
                    hiddenDelta.append(delta)
                deltas = [hiddenDelta]+deltas
            """Get output of all layers"""
            """
            Having aggregated all deltas, update the weights of the 
            hidden and output layers accordingly.
            """      
            for numLayer in xrange(0,self.numLayers):
                layer = self.layers[numLayer]
                for numNeuron in xrange(len(layer)):
                    weightMod = layer[numNeuron].updateWeights(allLayerOutput[numLayer], alpha, deltas[numLayer][numNeuron])
                    averageWeightChange += weightMod
                    numWeights += layer[numNeuron].inSize
            #end for each example
        #calculate final output
        averageError /= (len(examples)*len(examples[0][1]))             #number of examples x length of output vector
        averageWeightChange/=(numWeights)
        return averageError, averageWeightChange




def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008, hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet = startNNet
    """
    YOUR CODE
    """
    iteration = 0
    trainError = 0
    weightMod = 0

    weightChange = sys.maxint
    while iteration < maxItr and weightChange > weightChangeThreshold:
        trainError, weightChange = nnet.backPropLearning(examplesTrain, alpha)
        iteration = iteration + 1
        """
            Iterate for as long as it takes to reach weight modification threshold
        """
        # if iteration % 10 == 0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightChange)
        # else :
        #    print '.',

    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
	For each test example, you should first feedforward to get the NN outputs. Then, round the list of outputs from the output layer of the neural net.
	If the entire rounded list from the NN matches with the known list from the test example, then add to testCorrect, else add to testError.
    """ 

    testError = 0
    testCorrect = 0
    
    testAccuracy=0.0 # num correct / num total

    numTotal = len(examplesTest)
    for feature, label in examplesTest:
        # use updated weight to predict the output
        predict = [round(output) for output in nnet.feedForward(feature)[-1]]
        if predict == label:
            testCorrect += 1
        else:
            testError += 1
    testAccuracy = 1.0 * testCorrect / numTotal 

    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testCorrect,testError,(1 - testAccuracy) * 100)

    """return something"""
    return nnet, testAccuracy


