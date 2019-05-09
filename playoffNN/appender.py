import os
import numpy as np 

years = ['2016','2017','2019']

training_data = np.ndarray(shape=(90,5), dtype = "float")
training_labels = np.ndarray(shape=(30), dtype = "float")

fcounter = 0

for year in years:
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
				training_labels[idx] = np.float("1")
			else:
				training_labels[idx] = np.float("0")
			training_data[idx+(fcounter*30)][0] = np.float(items[7])
			training_data[idx+(fcounter*30)][1] = np.float(items[10])
			training_data[idx+(fcounter*30)][2] = np.float(items[13])
			training_data[idx+(fcounter*30)][3] = np.float(items[19])
			training_data[idx+(fcounter*30)][4] = np.float(items[22])
			print(training_data[idx])
	fcounter+=1

np.save("training_data", training_data)
np.save("training_labels", training_labels)
