#utility functions for neural net project
import random
def getNNPenData(fileString="datasets/pendigits.txt", limit=100000):
    """
    returns limit # of examples from penDigits file
    """
    examples=[]
    data = open(fileString)
    lineNum = 0
    for line in data:
        inVec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        outVec = [0,0,0,0,0,0,0,0,0,0]                      #which digit is output
        count=0
        for val in line.split(','):
            if count==16:
                outVec[int(val)] = 1
            else:
                inVec[count] = int(val)/100.0               #need to normalize values for inputs
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    return examples

def getList(num,length):
    list = [0]*length
    list[num-1] = 1
    return list
    
def getNNCarData(fileString ="datasets/car.data.txt", limit=100000):
    """
    returns limit # of examples from file passed as string
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['buying','maint','doors','persons','lug_boot','safety']
    attr_values = [['vhigh', 'high', 'med', 'low'],
                 ['vhigh', 'high', 'med', 'low'],
                 ['2','3','4','5more'],
                 ['2','4','more'],
                 ['small', 'med', 'big'],
                 ['high', 'med', 'low']]
    
    attrNNList = [('buying', {'vhigh' : getList(1,4), 'high' : getList(2,4), 'med' : getList(3,4), 'low' : getList(4,4)}),
                 ('maint',{'vhigh' : getList(1,4), 'high' : getList(2,4), 'med' : getList(3,4), 'low' : getList(4,4)}),
                 ('doors',{'2' : getList(1,4), '3' : getList(2,4), '4' : getList(3,4), '5more' : getList(4,4)}),
                 ('persons',{'2' : getList(1,3), '4' : getList(2,3), 'more' : getList(3,3)}),
                 ('lug_boot',{'small' : getList(1,3),'med' : getList(2,3),'big' : getList(3,3)}),
                 ('safety',{'high' : getList(1,3), 'med' : getList(2,3),'low' : getList(3,3)})]

    classNNList = {'unacc' : [1,0,0,0], 'acc' : [0,1,0,0], 'good' : [0,0,1,0], 'vgood' : [0,0,0,1]}
    
    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==6:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples


def buildExamplesFromPenData(size=10000):
    """
    build Neural-network friendly data struct
            
    pen data format
    16 input(attribute) values from 0 to 100
    10 possible output values, corresponding to a digit from 0 to 9

    """
    if (size != 10000):
        penDataTrainList = getNNPenData("datasets/pendigitsTrain.txt",int(.8*size))
        penDataTestList = getNNPenData("datasets/pendigitsTest.txt",int(.2*size))
    else :    
        penDataTrainList = getNNPenData("datasets/pendigitsTrain.txt")
        penDataTestList = getNNPenData("datasets/pendigitsTest.txt")
    return penDataTrainList, penDataTestList


def buildExamplesFromCarData(size=200):
    """
    build Neural-network friendly data struct
            
    car data format
    | names file (C4.5 format) for car evaluation domain

    | class values - 4 value output vector

    unacc, acc, good, vgood

    | attributes

    buying:   vhigh, high, med, low.
    maint:    vhigh, high, med, low.
    doors:    2, 3, 4, 5more.
    persons:  2, 4, more.
    lug_boot: small, med, big.
    safety:   low, med, high.
    """
    carData = getNNCarData()
    carDataTrainList = []
    for cdRec in carData:
        tmpInVec = []
        for cdInRec in cdRec[0] :
            for val in cdInRec :
                tmpInVec.append(val)
        #print "in :" + str(cdRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, cdRec[1])
        carDataTrainList.append(tmpList)
    #print "car data list : " + str(carDataList)
    tests = len(carDataTrainList)-size
    carDataTestList = [carDataTrainList.pop(random.randint(0,tests+size-t-1)) for t in xrange(tests)]
    return carDataTrainList, carDataTestList


def buildPotentialHiddenLayers(numIns, numOuts):
    """
    This builds a list of lists of hidden layer layouts
    numIns - number of inputs for data
    some -suggestions- for hidden layers - no more than 2/3 # of input nodes per layer, and
    no more than 2x number of input nodes total (so up to 3 layers of 2/3 # ins max
    """
    resList = []
    tmpList = []
    maxNumNodes = max(numOuts+1, 2 * numIns)
    if (maxNumNodes > 15):
        maxNumNodes = 15

    for lyr1cnt in range(numOuts,maxNumNodes):
        for lyr2cnt in range(numOuts-1,lyr1cnt+1):
            for lyr3cnt in range(numOuts-1,lyr2cnt+1):
                if (lyr2cnt == numOuts-1):
                    lyr2cnt = 0
                
                if (lyr3cnt == numOuts-1):
                    lyr3cnt = 0
                tmpList.append(lyr1cnt)
                tmpList.append(lyr2cnt)
                tmpList.append(lyr3cnt)
                resList.append(tmpList)
                tmpList = []
    return resList

def getNNExtraData(fileString ="datasets/nursery-data.txt", limit=100000):
    """
    returns limit # of examples from file passed as string
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['parents','has_nurs','form','children','housing','finance','social','health']
    attr_values = [['usual','pretentious','great_pret'],
                 ['proper','less_proper','improper','critical','very_crit'],
                 ['complete','completed','incomplete','foster'],
                 ['1','2','3','more'],
                 ['convenient','less_conv','critical'],
                 ['convenient','inconv'],
                 ['nonprob','slightly_prob','problematic'],
                 ['recommended','priority','not_recom']]

    attrNNList = [('parents', {'usual' : getList(1,3), 'pretentious' : getList(2,3), 'great_pret' : getList(3,3)}),
                     ('has_nurs',{'proper' : getList(1,5), 'less_proper' : getList(2,5), 'improper' : getList(3,5), 'critical' : getList(4,5), 'very_crit' : getList(5,5)}),
                     ('form',{'complete' : getList(1,4), 'completed' : getList(2,4), 'incomplete' : getList(3,4), 'foster' : getList(4,4)}),
                     ('children',{'1' : getList(1,4), '2' : getList(2,4), '3' : getList(3,4), 'more' : getList(4,4)}),
                     ('housing',{'convenient' : getList(1,3), 'less_conv' : getList(2,3), 'critical' : getList(3,3)}),
                     ('finance',{'convenient' : getList(1,2), 'inconv' : getList(2,2)}),
                     ('social',{'nonprob' : getList(1,3), 'slightly_prob' : getList(2,3), 'problematic' : getList(3,3)}),
                     ('health',{'recommended' : getList(1,3), 'priority' : getList(2,3), 'not_recom' : getList(3,3)})]    
    
    classNNList = {'not_recom' : [1,0,0,0,0], 'recommend' : [0,1,0,0,0], 'very_recom' : [0,0,1,0,0], 'priority' : [0,0,0,1,0], 'spec_prior' : [0,0,0,0,1]}
    
    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==8:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples

def buildExamplesFromExtraData(size=2500):
    """
    build Neural-network friendly data struct
            
    nursery data format
    | names file (C4.5 format) for nursery domain

    | class values

    not_recom, recommend, very_recom, priority, spec_prior

    | attributes

    parents:     usual, pretentious, great_pret.
    has_nurs:    proper, less_proper, improper, critical, very_crit.
    form:        complete, completed, incomplete, foster.
    children:    1, 2, 3, more.
    housing:     convenient, less_conv, critical.
    finance:     convenient, inconv.
    social:      nonprob, slightly_prob, problematic.
    health:      recommended, priority, not_recom.
    """
    extraData = getNNExtraData()
    extraDataTrainList = []
    for edRec in extraData:
        tmpInVec = []
        for edInRec in edRec[0] :
            for val in edInRec :
                tmpInVec.append(val)
        #print "in :" + str(edRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, edRec[1])
        extraDataTrainList.append(tmpList)
    #print "Nurery data list : " + str(extraDataList)
    tests = len(extraDataTrainList)-size
    extraDataTestList = [extraDataTrainList.pop(random.randint(0,tests+size-t-1)) for t in xrange(tests)]
    return extraDataTrainList, extraDataTestList



def getNNXorData(fileString ="datasets/xor-data.txt", limit=100000):
    """
    returns limit # of examples from file passed as string
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['a','b']
    attr_values = [['0','1'],
                 ['0','1']]

    attrNNList = [('a', {'0' : getList(1,2), '1' : getList(2,2)}),
                ('b',{'0' : getList(1,2), '1' : getList(2,2)})]    
    
    classNNList = {'0' : [1, 0], '1' : [0, 1]}
    
    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==2:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples

def buildExamplesFromXorData(size=4):
    xorData = getNNXorData()
    xorDataTrainList = []
    for xdRec in xorData:
        tmpInVec = []
        for xdInRec in xdRec[0] :
            for val in xdInRec :
                tmpInVec.append(val)
        #print "in :" + str(xdRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, xdRec[1])
        xorDataTrainList.append(tmpList)
    #print "XOR data list : " + str(xorDataList)
    # tests = len(xorDataTrainList)-size
    # xorDataTestList = [xorDataTrainList.pop(random.randint(0,tests+size-t-1)) for t in xrange(tests)]
    xorDataTestList = xorDataTrainList
    return xorDataTrainList, xorDataTestList
    