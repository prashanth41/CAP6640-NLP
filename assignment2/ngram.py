import numpy as np
import re
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score

with open('data/amazon_cells_labelled.txt','r+') as f:
	lines=[]
	for line in f:
		lines.append(line.lower())


for i in range(len(lines)):
	temp=lines[i].replace('\n','')
	lines[i]=temp

for i in range(len(lines)):
	lines[i]=lines[i].split("\t")

print(lines)

X=[]
y=[]

for i in range(len(lines)):
# for i in range(19):
	X.append(lines[i][0])
	y.append(lines[i][1])


tempX=[]

for i in range(len(X)):
	tempLine=""
	for j in range(len(X[i])):
		if X[i][j].isalpha()==True or X[i][j].isdigit()==True or X[i][j]==" ":
			tempLine+=X[i][j]

		else:
			if j+1<len(X[i]) and X[i][j]=='.':
				if X[i][j+1].isalpha()==True or X[i][j+1].isdigit()==True:
					tempLine+=" "




	tempX.append(tempLine)

X=tempX
print(X)

wordsMatrix=[]
for i in X:

	wordsMatrix.append(i.split())

print(wordsMatrix)


def feature_f1(X,wordsMatrix):

	# wordsMatrix=[]
	# for i in X:

	# 	wordsMatrix.append(i.split())

	words=[]

	for i in wordsMatrix:
		for j in i:
			words.append(j)


	uniqueWords=sorted(list(set(words)))
	# print(uniqueWords)
	print(len(uniqueWords))

	featureMatrix=np.zeros((1000,len(uniqueWords)),dtype=np.int)
	# print(featureMatrix)

	for i in range(len(wordsMatrix)):
		for j in range(len(wordsMatrix[i])):
			for k in range(len(uniqueWords)):
				if wordsMatrix[i][j]==uniqueWords[k]:
					featureMatrix[i][k]+=1

	# print(featureMatrix)


	return featureMatrix


# featureMatrix=feature_f1(X,wordsMatrix)
# print(wordsMatrix)



######Checking if the feature matrix is valid ################

# wordsMatrixSum=0
# for i in range(len(wordsMatrix)):
# 	for j in range(len(wordsMatrix[i])):
# 		wordsMatrixSum+=1

# print(wordsMatrixSum)

# featureMatrixSum=0
# for i in range(len(featureMatrix)):
# 	for j in range(len(featureMatrix[i])):
# 		featureMatrixSum+=featureMatrix[i][j]

# print(featureMatrixSum)

##############################################################





################# Training SVM for classification##############

X_train, X_test, y_train, y_test = train_test_split(featureMatrix, y, test_size = 0.10)
svclassifier = svm.SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('\n')
print(X_test[-1:])
print(svclassifier.predict(X_test[-1:]))


