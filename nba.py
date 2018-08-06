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

W1 = tf.Variable(tf.scalar_mul(0,tf.ones([1])))
W2 = tf.Variable(tf.scalar_mul(0,tf.ones([1])))
B = tf.Variable(tf.scalar_mul(0,tf.ones([1])))

PPG = tf.placeholder(tf.float32)
OPPG = tf.placeholder(tf.float32)
WINS = tf.placeholder(tf.float32)

WINS_pred = tf.add(tf.add((tf.multiply(W1,(PPG))),B),tf.multiply((W2),OPPG))
cost = tf.log(tf.reduce_sum(tf.square(tf.subtract(WINS,WINS_pred))))

learning_rate = .0005
epochs = 30000
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

log = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	statsArr = np.asarray(stats)
	min_cost = 9000
	for epoch in range(0,epochs):
		for team in statsArr:
			sess.run(optimizer, feed_dict={PPG:team[1],OPPG:team[2],WINS:team[14]})
		training_cost = sess.run(cost,feed_dict={PPG:statsArr[:,1],OPPG:statsArr[:,2],WINS:statsArr[:,14]})
		#log.append(training_cost)
		#print(training_cost)
		if training_cost < min_cost:
			print("old",min_cost,"new",training_cost)
			min_cost=training_cost
			w1 = W1.eval()
			w2 = W2.eval()
			b = B.eval()
			print(w1)
			print(w2)
		'''
		if(training_cost<12):
			print("training cost",training_cost)
			for i in range(0,30):
				print((statsArr[i][1]*a*statsArr[i][1])+b,"vs",statsArr[i][14])
			print(a,b,"weights")
		'''
		

'''
#plot the cost
import matplotlib.pyplot as plt
plt.axis([0, epochs, 0, 20])
for i in range(0,epochs):
	plt.plot(i,log[i],'.')
plt.show()
'''
print("final weights are ", w1, w2)

predictions = []
wins = []

for i in range(0,30):
	print("Weight",w1,"times",statsArr[i][1],"plus weight",w2,"times",statsArr[i][2],"plus", b, "equals",(statsArr[i][1]*w1)+(w2*statsArr[i][2])+b,"vs",statsArr[i][14])
	predictions.append((statsArr[i][1]*w1)+(w2*statsArr[i][2])+b)
	wins.append(statsArr[i][14])

#predictions vs wins
norm_preds = [i/sum(predictions) for i in predictions]
norm_wins = [i/sum(wins) for i in wins]
import matplotlib.pyplot as plt

plt.axis([0, 30, 0, .1])
for i in range(0,30):
	plt.plot(i,norm_preds[i],'ro')
	plt.plot(i,norm_wins[i],'b^')
plt.show()


'''
import matplotlib.pyplot as plt
plt.axis([0, 30, 20, 68])
for i in range(0,30):
	plt.plot(i,predictions[i],'ro')
	plt.plot(i,wins[i],'b^')
plt.show()
'''

'''
import matplotlib.pyplot as plt
plt.axis([0, 30, 0, 120])
for i in range(0,29):
	plt.plot(i,statsArr[i][1],'ro')
	plt.plot(i,statsArr[i][14],'b^')
	print(statsArr[i][0],statsArr[i][14])
plt.show()

'''