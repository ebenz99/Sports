#bringing in the stats
import csv
import numpy as np 
import random

infile = open('NBAcsvESPN.csv')
CSV_file = csv.reader(infile)

numToTeam = {}
stats = []

for idx,row in enumerate(CSV_file):
	if idx>1:
		numToTeam[row[0]] = row[1]
	else:
		continue
	insertIdx = 0
	teamStats = np.array([])
	for colIdx,item in enumerate(row):
		if colIdx != 1:
			itm = float(item)
			teamStats = np.insert(arr=teamStats,obj=insertIdx,values=itm)
			insertIdx+=1
	stats.append(teamStats)


winfile = open('Wins.csv')
CSV_winfile = csv.reader(winfile)

teamToWin = {}
for idx,row in enumerate(CSV_winfile):
	teamToWin[row[0]] = row[1]


for idx,team in enumerate(stats):
	win = teamToWin[numToTeam[str(int(team[0]))]]
	stats[idx] = np.insert(arr=stats[idx],obj=14,values=win)



#learning part
import tensorflow as tf 

W1 = tf.Variable(tf.scalar_mul(2,tf.ones([1])))

PPG = tf.placeholder(tf.float32)
WINS = tf.placeholder(tf.float32)

WINS_pred = (tf.multiply(W1,tf.square(PPG)))
cost = tf.log(tf.reduce_sum(tf.square(tf.subtract(WINS,WINS_pred))))

learning_rate = .005
epochs = 1000
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

log = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	statsArr = np.asarray(stats)
	for epoch in range(0,epochs):
		for team in statsArr:
			sess.run(optimizer, feed_dict={PPG:team[1],WINS:team[14]})
		training_cost = sess.run(cost,feed_dict={PPG:statsArr[:,1],WINS:statsArr[:,14]})
		log.append(training_cost)
		#print(training_cost)
	a = W1.eval()


#plot the cost
import matplotlib.pyplot as plt
plt.axis([0, epochs, 0, 20])
for i in range(0,epochs):
	plt.plot(i,log[i],'.')
plt.show()

print("final weight is ", a)

for i in range(0,30):
	print("Weight",a,"times",statsArr[i][1],"squared equals",(a*statsArr[i][1]*a*statsArr[i][1]),"vs",statsArr[i][14])

'''
import matplotlib.pyplot as plt
plt.axis([0, 30, 0, 120])
for i in range(0,29):
	plt.plot(i,statsArr[i][1],'ro')
	plt.plot(i,statsArr[i][14],'b^')
	print(statsArr[i][0],statsArr[i][14])
plt.show()

'''