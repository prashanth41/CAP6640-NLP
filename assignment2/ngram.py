import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


with open('data/imdb_labelled.txt','r+') as f:
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
# for i in range(20):
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

for i in range(len(y)):
	y[i]=int(y[i])

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




def features(X,wordsMatrix,n):
	grams=[]

	for i in range(len(wordsMatrix)):
		for j in range(len(wordsMatrix[i])-(n-1)):
			temp=''
			for k in range(n):
				temp+=wordsMatrix[i][j+k]
			grams.append(temp)

	print(grams)

	uniqueGrams=sorted(list(set(grams)))
	print(uniqueGrams)

	featureMatrix=np.zeros((len(wordsMatrix),len(uniqueGrams)),dtype=np.int)

	for i in range(len(wordsMatrix)):
		for j in range(len(wordsMatrix[i])-(n-1)):
			temp=''
			for k in range(n):
				temp+=wordsMatrix[i][j+k]

			for l in range(len(uniqueGrams)):
				if temp==uniqueGrams[l]:
					featureMatrix[i][l]+=1

	print(featureMatrix)
	print(grams)
	print(uniqueGrams)
	print(len(uniqueGrams))
	print(len(grams))




	return featureMatrix


featureMatrix=features(X,wordsMatrix,3)

featuresSum=0
for i in range(len(featureMatrix)):
	for j in range(len(featureMatrix[i])):
		featuresSum+=featureMatrix[i][j]

print(featuresSum)



#################10-fold cross valiadtion###############

Xpos=[]
Xneg=[]
ypos=[]
yneg=[]

for i in range(len(y)):
	if y[i]==0:
		yneg.append(y[i])
		Xneg.append(featureMatrix[i])

	elif y[i]==1:
		ypos.append(y[i])
		Xpos.append(featureMatrix[i])

# print(len(ypos))
# print(len(Xneg))


def classifiy(X_train,y_train,X_test,y_test):

	svclassifier = svm.SVC(kernel='linear')  
	svclassifier.fit(X_train, y_train) 
	y_pred = svclassifier.predict(X_test)

	print(confusion_matrix(y_test,y_pred)) 
	confusionMatrix=confusion_matrix(y_test,y_pred) 
	# print(classification_report(y_test,y_pred))
	print(accuracy_score(y_test,y_pred))
	precision=precision_score(y_test, y_pred)
	recall=recall_score(y_test, y_pred)
	accuracy=accuracy_score(y_test,y_pred)
	# accuracy+=accuracy_score(y_test,y_pred)

	return accuracy,precision,recall,confusionMatrix



foldLength=len(ypos)//10
print(foldLength)

accuracyList=[]
confusionMatrixSum=np.zeros((2,2),dtype=np.int)
precisionList=[]
recallList=[]



for i in range(10):

	print("for fold = %d" %(i+1))

	X_train=[]
	y_train=[]
	X_test=[]
	y_test=[]


	if i==0:

		X_test+=(Xpos[:(i+1)*foldLength])
		X_test+=(Xneg[:(i+1)*foldLength])

		y_test+=(ypos[:(i+1)*foldLength])
		y_test+=(yneg[:(i+1)*foldLength])

		X_train+=(Xpos[(i+1)*foldLength:])
		X_train+=(Xneg[(i+1)*foldLength:])

		y_train+=(ypos[(i+1)*foldLength:])
		y_train+=(yneg[(i+1)*foldLength:])


	elif i==9:

		X_test+=(Xpos[foldLength*9:])
		X_test+=(Xneg[foldLength*9:])

		y_test+=(ypos[foldLength*9:])
		y_test+=(yneg[foldLength*9:])

		X_train+=(Xpos[:foldLength*9])
		X_train+=(Xneg[:foldLength*9])

		y_train+=(ypos[:foldLength*9])
		y_train+=(yneg[:foldLength*9])


	else:



		X_test+=(Xpos[i*foldLength:(i+1)*foldLength])
		X_test+=(Xneg[i*foldLength:(i+1)*foldLength])

		y_test+=(ypos[i*foldLength:(i+1)*foldLength])
		y_test+=(yneg[i*foldLength:(i+1)*foldLength])

		X_train+=(Xpos[0:i*foldLength])
		X_train+=(Xpos[(i+1)*foldLength:])
		X_train+=(Xneg[0:i*foldLength])
		X_train+=(Xneg[(i+1)*foldLength:])

		y_train+=(ypos[0:i*foldLength])
		y_train+=(ypos[(i+1)*foldLength:])
		y_train+=(yneg[0:i*foldLength])
		y_train+=(yneg[(i+1)*foldLength:])


	X_train=np.array(X_train)
	y_train=np.array(y_train)
	X_test=np.array(X_test)
	y_test=np.array(y_test)

	# print(i)
	# print(len(X_train))
	# print(len(X_test))
	# print(len(y_train))
	# print(len(y_test))
	# print(len(ypos))
	# print(len(Xneg))



	# print(len(Xpos[0]))
	# print(len(X_test[0]))

################# Training SVM for classification##############


	accuracy,precision,recall,confusionMatrix=classifiy(X_train,y_train,X_test,y_test)

	accuracyList.append(accuracy)
	precisionList.append(precision)
	recallList.append(recall)

	for i in range(len(confusionMatrix)):
		for j in range(len(confusionMatrix[i])):
			confusionMatrixSum[i][j]+=confusionMatrix[i][j]

	print('\n')
	# print(X_test[-1:])
	# print(svclassifier.predict(X_test[-1:]))

print(np.sum(accuracyList)/10)
print(np.sum(precisionList)/10)
print(np.sum(recallList)/10)
print(confusionMatrixSum)

