import os
import numpy as np 

training_years = ['2016','2017','2019']
testing_years = ['2018']


training_data = np.ndarray(shape=(90,5,1), dtype = "float")
training_labels = np.ndarray(shape=(90), dtype = "float")

fcounter = 0

for year in training_years:
	''' code for getting wins
	dic = {}
	with open(year+"Wins.csv","r") as winfile:
		for line in winfile:
			items = line.split(",")
			dic[items[1]] = items[3]
	'''
	with open(year+"Stats.csv","r") as statfile:
		for idx,line in enumerate(statfile):
			items = line.split(",")
			#if items[1] not in dic:
			#	print(items[1])
			#	print("FAILED")
			#	exit(2)
			#else:
			if "*" in items[1]:
				training_labels[idx+(fcounter*30)] = np.float("1")
			else:
				training_labels[idx+(fcounter*30)] = np.float("0")
			training_data[idx+(fcounter*30)][0][0] = np.float(items[7])
			training_data[idx+(fcounter*30)][1][0] = np.float(items[10])
			training_data[idx+(fcounter*30)][2][0] = np.float(items[13])
			training_data[idx+(fcounter*30)][3][0] = np.float(items[19])
			training_data[idx+(fcounter*30)][4][0] = np.float(items[22])
	fcounter+=1

np.save("training_data", training_data)
np.save("training_labels", training_labels)


testing_data = np.ndarray(shape=(30,5,1), dtype = "float")
testing_labels = np.ndarray(shape=(30), dtype = "float")

fcounter = 0

for year in testing_years:
	''' code for getting wins
	dic = {}
	with open(year+"Wins.csv","r") as winfile:
		for line in winfile:
			items = line.split(",")
			dic[items[1]] = items[3]
	'''
	with open(year+"Stats.csv","r") as statfile:
		for idx,line in enumerate(statfile):
			items = line.split(",")
			#if items[1] not in dic:
			#	print(items[1])
			#	print("FAILED")
			#	exit(2)
			#else:
			if "*" in items[1]:
				testing_labels[idx+(fcounter*30)] = np.float("1")
			else:
				testing_labels[idx+(fcounter*30)] = np.float("0")
			testing_data[idx+(fcounter*30)][0][0] = np.float(items[7])
			testing_data[idx+(fcounter*30)][1][0] = np.float(items[10])
			testing_data[idx+(fcounter*30)][2][0] = np.float(items[13])
			testing_data[idx+(fcounter*30)][3][0] = np.float(items[19])
			testing_data[idx+(fcounter*30)][4][0] = np.float(items[22])
	fcounter+=1

np.save("testing_data", testing_data)
np.save("testing_labels", testing_labels)
