from math import log
import sys
from scipy.stats import chi2
from scipy.stats import chisqprob

class Node:
  """
  A simple node class to build our tree with. It has the following:
  
  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by. 
  isleaf (boolean): whether this is a leaf. False.
  """
  
  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.
    
    value (str): Since this is a leaf node, a final value for the label.
    isleaf (boolean): whether this is a leaf. True.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True
    
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)
    
  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string    

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count  

  def __str__(self):
    return self.preorder(0, self.root)
  
  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`
    
    Args:
        classificationData (dictionary<string,string>): dictionary of attribute values
    Returns:
        str
        The classification made with this tree.
    """

    # print classificationData # example will be read from carsEntropy.test one line each classify

    # if self.root.isleaf:
    #     # final leaf default label
    #     return self.root.value

    currentNode = self.root
    while currentNode.isleaf is False:
        """
        To see if example can be classify to that branch with 'key' value.
        e.g: safety has three branch: high, med, low
        if the input example matches any one of those branch, then it will
        be classified to this branch and keep going down the tree utill it
        reach the leaf node.
        """
        for key, value in currentNode.children.iteritems():
            # print k, " + ", v, " + ", len(node.children)
            # print node.attr, " + ", classificationData[node.attr]
            if key == classificationData[currentNode.attr]:
                # new current node
                currentNode = currentNode.children[key]
                break
    return currentNode.value    
  
def getPertinentExamples(examples,attrName,attrValue):
    """
    Helper function to get a subset of a set of examples for a particular assignment 
    of a single attribute. That is, this gets the list of examples that have the value 
    attrValue for the attribute with the name attrName.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValue (str): a value of the attribute
    Returns:
        list<dictionary<str,str>>
        The new list of examples.
    """
    newExamples = []
    # print examples
    # print attrName
    # print attrValue

    for example in examples:
        if example[attrName] == attrValue:
            newExamples.append(example)
    return newExamples
  
def getClassCounts(examples, className):
    """
    Helper function to get a dictionary of counts of different class values
    in a set of examples. That is, this returns a dictionary where each key 
    in the list corresponds to a possible value of the class and the value
    at that key corresponds to how many times that value of the class 
    occurs.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        className (str): the name of the class
    Returns:
        dictionary<string,int>
        This is a dictionary that for each value of the class has the count
        of that class value in the examples. That is, it maps the class value
        to its count.
    """
    classCounts = {}
    # print className
    # print examples

    for example in examples:
        if example[className] in classCounts:
            classCounts[example[className]] += 1
        else:
            classCounts[example[className]] = 1
    return classCounts

def getMostCommonClass(examples, className):
    """
    A freebie function useful later in makeSubtrees. Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples, attrName, attrValues, className):
    """
    Helper function to get a dictionary of counts of different class values
    corresponding to every possible assignment of the passed in attribute. 
	  That is, this returns a dictionary of dictionaries, where each key  
	  corresponds to a possible value of the attribute named attrName and holds
 	  the counts of different class values for the subset of the examples
 	  that have that assignment of that attribute.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<str>): list of possible values for the attribute
        className (str): the name of the class
    Returns:
        dictionary<str,dictionary<str,int>>
        This is a dictionary that for each value of the attribute has a
        dictionary from class values to class counts, as in getClassCounts
    """
    attributeCounts = {}
    # print attrName
    # print attrValues
    # print className

    # for example in examples: # this wrong line makes connect4 dataset running like endlessly
    for attrValue in attrValues:
        newExamples = getPertinentExamples(examples, attrName, attrValue)
        classCounts = getClassCounts(newExamples, className)
        attributeCounts[attrValue] = classCounts
    return attributeCounts
        

def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    This is called H in the book. Note that our labels are not binary,
    so the equations in the book need to be modified accordingly. Note
    that H is written in terms of B, and B is written with the assumption 
    of a binary value. B can easily be modified for a non binary class
    by writing it as a summation over a list of ratios, which is what
    you need to implement.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The set entropy score of this list of class value counts.
    """
    # print classCounts

    entropy = 0.0
    # convert int to float
    total = sum(classCounts) * 1.0
    for count in classCounts:
        # math domain error
        entropy += -1.0 * (count / total) * (log(count / total, 2))
    return entropy

def remainder(examples, attrName, attrValues, className):
    """
    Calculates the remainder value for given attribute and set of examples.
    See the book for the meaning of the remainder in the context of info 
    gain.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The remainder score of this value assignment of the attribute.
    """

    # print className
    # print attrName

    remainder = 0.0
    totalExamples = len(examples) * 1.0
    classCount = []
    # {attrValue: {className: count}} dictionary<str,dictionary<str,int>>
    attributeCounts = getAttributeCounts(examples, attrName, attrValues, className)
    for attrValue, classCountDict in attributeCounts.iteritems():
        if len(classCountDict) != 0:
            classCount = classCountDict.values()
            attributeEntropy = setEntropy(classCount)
            totalAttribute = sum(classCount)
            remainder += totalAttribute / totalExamples * attributeEntropy
    return remainder

def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    See the book for the equation - it's a combination of setEntropy and
    remainder (setEntropy replaces B as it is used in the book).
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The gain score of this value assignment of the attribute.
    """

    # convert dictionary values into a list of values
    classCounts = getClassCounts(examples, className).values()
    entropy = setEntropy(classCounts)
    remain = remainder(examples, attrName, attrValues, className) 
    return entropy - remain 

def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    See equation in instructions.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The gini score of this list of class value counts.
    """

    giniScore = 1.0
    totalExamples = sum(classCounts) * 1.0
    for count in classCounts:
        giniScore -= (count / totalExamples) ** 2
    return giniScore
   
def giniGain(examples, attrName, attrValues, className):
    """
    Return the inverse of the giniD function described in the instructions.
    The inverse is returned so as to have the highest value correspond 
    to the highest information gain as in entropyGain. If the sum is 0,
    return sys.maxint.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The summed gini index score of this list of class value counts.
    """

    giniScoreGain = 0.0
    totalExamples = len(examples) * 1.0
    classCount = []
    attributeCounts = getAttributeCounts(examples, attrName, attrValues, className)
    for attrValue, classCountDict in attributeCounts.iteritems():
        if len(classCountDict) != 0:
            classCount = classCountDict.values()
            giniScore = giniIndex(classCount)
            totalAttribute = sum(classCount)
            giniScoreGain += totalAttribute / totalExamples * giniScore
    if giniScoreGain == 0.0:
        return sys.maxint
    return 1 / giniScoreGain   

    
def makeTree(examples, attrValues, className, setScoreFunc, gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes = attrValues.keys()
    return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc))
    
def makeSubtrees(remainingAttributes, examples, attributeValues, className, defaultLabel, setScoreFunc, gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    # print remainingAttributes # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print examples # [{0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 'label': 0}, ...
    # print attributeValues # {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0, 1], 5: [0, 1], 6: [0, 1], 7: [0, 1], 8: [0, 1], 9: [0, 1]}
    # print className # label
    # print defaultLabel # 0
    # print setScoreFunc
    # print gainFunc

    def sameClassification(examples, className):
            result = [False, None]
            classCounts = getClassCounts(examples, className)
            for key, value in classCounts.iteritems():
                if value == len(examples):
                    result[0] = True
                    result[1] = key
            return result

    if len(examples) == 0:
        return LeafNode(defaultLabel)
    elif sameClassification(examples, className)[0]:
        return LeafNode(sameClassification(examples, className)[1])
    elif len(remainingAttributes) == 0:
        return LeafNode(getMostCommonClass(examples, className))

    maxAttrName = remainingAttributes[0]
    maxGain = gainFunc(examples, maxAttrName, attributeValues[maxAttrName], className)
    for attrName in remainingAttributes[1 : ]:
        gain = gainFunc(examples, attrName, attributeValues[attrName], className)
        if gain > maxGain:
            maxGain = gain
            maxAttrName = attrName

    rootNode = Node(maxAttrName)
    # deep copy
    newRemainingAttributes = list(remainingAttributes)
    newRemainingAttributes.remove(maxAttrName)
    defaultLabel = getMostCommonClass(examples, className)

    for attrValue in attributeValues[maxAttrName]:
        newExamples = getPertinentExamples(examples, maxAttrName, attrValue)
        childNode = makeSubtrees(newRemainingAttributes, newExamples, attributeValues, className, defaultLabel, setScoreFunc, gainFunc)
        rootNode.children[attrValue] = childNode
    return rootNode


def makePrunedTree(examples, attrValues, className, setScoreFunc, gainFunc, q):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes = attrValues.keys()
    return Tree(makePrunedSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc,q))
    
def makePrunedSubtrees(remainingAttributes, examples, attributeValues, className, defaultLabel, setScoreFunc, gainFunc, q):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie classEntropy or gini)
        gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """

#################################################################
    def sameClassification(examples, className):
            result = [False, None]
            classCounts = getClassCounts(examples, className)
            for key, value in classCounts.iteritems():
                if value == len(examples):
                    result[0] = True
                    result[1] = key
            return result

#################################################################
    def devX(examples, className, attributeValues, maxAttrName):
        dev = 0.0
        classCountDict = getClassCounts(examples, className)
        classCountList = classCountDict.values() # [p, n]
        dTotal = sum(classCountList) # (p + n)

        for attrValue in attributeValues[maxAttrName]:
            newExamples = getPertinentExamples(examples, maxAttrName, attrValue)
            xClassCountDict = getClassCounts(newExamples, className)
            xClassCountList = xClassCountDict.values() # [px, nx]
            dxTotal = sum(xClassCountList) # |Dx| = px + nx

            for key in xClassCountDict.keys():
                pX = xClassCountDict[key] * 1.0
                pX_ = classCountDict[key] * 1.0 / dTotal * dxTotal # p / (p + n) * |Dx|
                dev += (pX - pX_) ** 2 / pX_
        return dev
#################################################################

    if len(examples) == 0:
        return LeafNode(defaultLabel)
    elif sameClassification(examples, className)[0]:
        return LeafNode(sameClassification(examples, className)[1])
    elif len(remainingAttributes) == 0:
        return LeafNode(getMostCommonClass(examples, className))

    maxAttrName = remainingAttributes[0]
    maxGain = gainFunc(examples, maxAttrName, attributeValues[maxAttrName], className)
    for attrName in remainingAttributes[1 : ]:
        gain = gainFunc(examples, attrName, attributeValues[attrName], className)
        if gain > maxGain:
            maxGain = gain
            maxAttrName = attrName

    dev = devX(examples, className, attributeValues, maxAttrName)
    # print dev, " + ", dev1
    v = len(attributeValues[maxAttrName])

    if chi2.sf(dev, v - 1) > q:
        return LeafNode(getMostCommonClass(examples, className))

    rootNode = Node(maxAttrName)
    # deep copy
    newRemainingAttributes = list(remainingAttributes)
    newRemainingAttributes.remove(maxAttrName)

    defaultLabel = getMostCommonClass(examples, className)
    for attrValue in attributeValues[maxAttrName]:
        newExamples = getPertinentExamples(examples, maxAttrName, attrValue)
        childNode = makePrunedSubtrees(newRemainingAttributes, newExamples, attributeValues, className, defaultLabel, setScoreFunc, gainFunc, q)
        rootNode.children[attrValue] = childNode
    return rootNode