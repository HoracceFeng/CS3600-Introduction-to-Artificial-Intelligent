from DataInterface import getExtraCreditDataset
from DecisionTree import makeTree, setEntropy, infoGain
from Testing import *

def testExtraCredit(setFunc = setEntropy, infoFunc = infoGain):
    examples,attrValues,labelName,labelValues = getExtraCreditDataset()
    print 'Testing Nursery dataset. Number of examples %d.'%len(examples)
    tree = makeTree(examples, attrValues, labelName, setFunc, infoFunc)
    f = open('nursery.out','w')
    f.write(str(tree))
    f.close()
    print 'Tree size: %d.\n'%tree.count()
    print 'Entire tree written out to nursery.out in local directory\n'
    dataset = getExtraCreditDataset()
    evaluation = getAverageClassificaionRate((examples,attrValues,labelName,labelValues),runs=10,testSize=2000)
    print 'Results for training set:\n%s\n'%str(evaluation)
    printDemarcation()
    return (tree,evaluation)

def main():
    testExtraCredit()

if __name__=='__main__':
    main()