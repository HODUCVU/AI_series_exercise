import csv
import numpy as np
import math
from collections import Counter

def loadData(path, K = 1):
    with open(path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(list(data))
    data = np.delete(data, 0, 0) # delete header
    data = np.delete(data, 0, 1) # delete id 
    data = np.array(data, dtype=float)  # Convert all data to float
    file.close()
    # Convert the data to float, except for the last two columns which are integer
    data[:, -2:] = data[:, -2:].astype(int)
    testset = data[data[:, -1] == K]
    dset = [data[data[:, -1] == i] for i in range(1, 11) if i != K]
    return dset, testset

def distancs_n2(pointA, pointB, numberOfFeature = 4):
    sum = 0
    for i in range(numberOfFeature):
        tmp = pointA[i] - pointB[i]
        tmp = tmp ** 2
        sum = np.sum(tmp)
    return np.sqrt(sum)

def distancs_n1(pointA, pointB, numberOfFeature = 4):
    tmp = 0
    for i in range(numberOfFeature):
        tmp += abs(float(float(pointA[i]) - float(pointB[i])))
    return tmp;

def KNN(dset, point, k):
    distancs = []
    for item in dset:
        distancs.append({
            "label": item[-2],
            "value": distancs_n2(item, point)
        })
    distancs.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distancs]
    return labels[:k]

def findMostOccur(arr):
    labels = set(arr); # set label
    answer = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            answer = label
    return answer   

if __name__ == '__main__':
    dset, testset = loadData(path='./Iris.csv',K=2)
    k = 3
    for item in testset:
        knn = KNN(dset=dset, point=item, k=k)
        answer = findMostOccur(knn)
        print("label: {} -> predicted: {}".format(item[-2], answer))
