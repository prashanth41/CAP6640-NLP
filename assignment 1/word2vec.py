import math
import numpy as np 
import sys



file = sys.argv[1]
windowSize = int(sys.argv[2])

with open(file,'r') as f:
	data=f.read()

data = data.lower()
# print(data)

words = data.split(' ')
#print(words)

uniqueWords = list(set(words))
uniqueWords = sorted(uniqueWords)
# print(uniqueWords)
paddedUniqueWords = ['#'] + uniqueWords 

# add padding of window size at start and end
padding = ['#']*windowSize
# print(padding)

paddedData = padding + words
paddedData = paddedData + padding


print(paddedData)
# print(uniqueWords)
# print(paddedUniqueWords)

coOccurrence = {}


# print(paddedDict)


for i in uniqueWords:

	coOccurrence[i]={}
	for j in paddedUniqueWords:

		coOccurrence[i][j]=0


# print(coOccurrence)


def windowCount(windowSize,wordIndex,word):

	for i in range(wordIndex - windowSize, wordIndex):
		# print([paddedData[i]])
		coOccurrence[word][paddedData[i]]+= 1

	for i in range(wordIndex+1, (wordIndex+windowSize)+1):
		# print(coOccurrence[word][paddedData[i]])
		coOccurrence[word][paddedData[i]]+= 1

	return 0


for word in uniqueWords:

	# print(word)

	wordIndices = [i for i,x in enumerate(paddedData) if x==word]
	print(wordIndices)

	for j in wordIndices:

		print(j)

		windowCount(windowSize,j,word)

		# print(coOccurrence)


print(coOccurrence)


def normalize(key):

	tempSum = sum(coOccurrence[key].values())
	for i in coOccurrence[key]:
		coOccurrence[key][i] = coOccurrence[key][i]/(tempSum)

	return 0

for i in coOccurrence:
	normalize(i)

print(coOccurrence)


#removing padding

for i in coOccurrence:

	del coOccurrence[i]['#']

print(coOccurrence)

coOccurrenceList = []  

for i in coOccurrence:
	coOccurrenceList.append(list(coOccurrence[i].values()))

# print(coOccurrenceList)

coOccurrenceArray = np.array([np.array(i) for i in coOccurrenceList])
print(coOccurrenceArray)


la = np.linalg
U, s , V = la.svd(coOccurrenceArray, full_matrices=False)
print(U)

# word2vec = 