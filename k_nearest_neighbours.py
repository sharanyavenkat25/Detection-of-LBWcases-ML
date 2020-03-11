import csv
import random
import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import math
from datetime import datetime

def data_cleaning(data):
    ''' Function: Clean data set by replacing missing
        values with median/mode values as required
        Arguments: dataset 
        Returns: cleaned dataset
    '''
    median1 = data['age'].median()
    data['age'].fillna(median1, inplace=True)
    median2 = data['BP1'].median()
    data['BP1'].fillna(median2, inplace=True)
    mean1 = data['weight1'].mean()
    data['weight1'].fillna(mean1, inplace=True)
    mode1 = data['education'].mode()
    data['education'].fillna(mode1[0], inplace=True)
    mode2 = data['res'].mode()
    data['res'].fillna(mode2[0], inplace=True)
    mode3 = data['history'].mode()
    data['history'].fillna(mode3[0], inplace=True)
    median3 = data['HB'].median()
    data['HB'].fillna(median3, inplace=True)
    return data


def euclidean(test, train, length):
    '''Function: Computes euclidean distance between test and train
        Arguments: (1) test - dataset
                   (2) train-dataset
                   (3) length- no of rows in test data set
        Returns: euclidean distance
    '''

    distance = 0
    for x in range(length):
    	distance = distance + (test[x] - train[x])*(test[x]-train[x])
    return math.sqrt(distance)

def find_nearest_neigbours(k,test_inst,train):
    '''Function: Finds nearest neighbours of each test instance passed
        Arguments: k(int) - no. of neighbours
                   test_inst- one row of test data set
                   train- train dataset
        Returns: List of nearest neighbours sorted by distance
    '''
    distances=dict()
    for x in range(len(train)):

        dist = euclidean(test_inst, train[x], len(test_inst)-1)
        distances[x]=dist
    distances=sorted(distances.items(), key = lambda kv:kv[1])
    neighbours = []
    rows=[]
    for i in distances:
        rows.append(i[0])
     
    for x in rows[0:k]:
        neighbours.append(train[x])
    return neighbours

def results(neighbours,k):
    '''Function: Calculates the majority label of nearest neighbours
       Arguments: neighbors - list of neighbours
                  k - no. of neighbours
       Returns: majority label [1/0]
    '''
    count_0=0
    count_1=0
    for i in range(k):
        #Label is the last column of each of the neighbours
        label=neighbours[i][-1]
        if label==1:
            count_1+=1
        else:
            count_0+=1
    #Checking for majority labels
    if(count_1>count_0):
        return 1
    else:
        return 0
        

#if __name__ == '__main__':
def main(k):   
    trainingSet=[]
    testSet=[]
    #Read the dataset
    data = pd.read_csv('Andhra_dataset2.csv') 
    #Clean the data
    data_cleaning(data)
    #splitting into test and train sets
    trainingSet, testSet = train_test_split(data,test_size = 0.10,random_state=27)
    #0.10,27
    #Initialize no. of neighbours
    sc = SC()
    no_of_neighbours= k 
    testSet=np.array(testSet)
    trainingSet=np.array(trainingSet)
    #Normalize the data set
    testSet[:,0:9]=sc.fit_transform(testSet[:,0:9])
    trainingSet[:,0:9]=sc.fit_transform(trainingSet[:,0:9])
    len_testSet=len(testSet)

    count=0
    cm=[[0,0],[0,0]]
    tp=0
    tn=0
    fp=0
    fn=0
    startTime = datetime.now().microsecond
    for i in range(len_testSet):
        #Finds nearest  neighbours for each test instance
        neighbours = find_nearest_neigbours(no_of_neighbours,testSet[i],trainingSet)
        #results for that test instance 
        result = results(neighbours,no_of_neighbours)
        #if the predicted label and the actual label are the same, count is incremented
        if(result-testSet[i][-1]==0):
            if(result==0):
                tn+=1
            else:
                tp+=1
            count=count+1
        else:
            if(result==1):
                fp+=1
            else:
                fn+=1
    
    #Accuracy of the model is printed
    cm[0][0]=tp
    cm[0][1]=fn
    cm[1][0]=fp
    cm[1][1]=tn

    endTime=datetime.now().microsecond
    time=(endTime - startTime)/1000000.0

    # Results
    print(f"\n##########    k-Nearest Neighbours to detect Potential Low Birth Weight Cases   ##########\n")
    print(f"Model Parameter :\n \t Value of k:{k}\n")
    print(f"Time taken for Evaluation: {time} seconds\n")
    print("Confusion Matrix : \n")
    print(cm)
    print("\nRESULTS \n")
    print ("\tAccuracy percentage: ", count/float(len(testSet))*100)
    print("\tAccuracy:",(tp+tn)/(tp+tn+fp+fn))
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    fscore=(2*p*r)/(p+r)
    print("\tPrecision:",tp/(tp+fp))
    print("\tRecall:",tp/(tp+fn))
    print("\tf1 score: ",fscore)
    #return (count/float(len(testSet))*100)

main(3)





# In[89]:

'''
now=dict()
for i in range(60):
    now[i]=main(i)
import matplotlib.pyplot as plt
plt.plot(now.keys(),now.values())
plt.xlabel('No. of Neighbours') 
# naming the y axis 
plt.ylabel('Accuracy')'''


