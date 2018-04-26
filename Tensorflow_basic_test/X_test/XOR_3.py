import tensorflow as tf
import numpy as np
import sys


isTrain = input('是否訓練(Y/N)?')

if isTrain == 'Y' or isTrain == 'y':

	X = np.loadtxt('Train_data_X.txt', delimiter=',')
	Y = np.loadtxt('Train_data_Y.txt', delimiter=',')
	Y = np.reshape(Y,(len(Y),1))
	
else:
	
	X = np.loadtxt('input_data_X.txt', delimiter=',')



N_STEPS = 250001
N_EPOCH = 250000
N_TRAINING = len(X)

N_INPUT_NODES = 4
N_HIDDEN_NODES = 10
N_OUTPUT_NODES  = 1
ACTIVATION = 'tanh' # sigmoid or tanh
COST = 'ACE' # MSE or ACE
LEARNING_RATE = 0.001

if __name__ == '__main__':

	x_ = tf.placeholder(tf.float32, shape=[None, N_INPUT_NODES], name="x-input")
	y_ = tf.placeholder(tf.float32, shape=[None, N_OUTPUT_NODES], name="y-input")

	theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1, 1), name="theta1")
	theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_OUTPUT_NODES], -1, 1), name="theta2")

	bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
	bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")
	
	if ACTIVATION == 'sigmoid':

		### Use a sigmoidal activation function ###
		layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
		output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

	else:

		### Use tanh activation function ###

		layer1 = tf.tanh(tf.matmul(x_, theta1) + bias1)
		output = tf.tanh(tf.matmul(layer1, theta2) + bias2)

		output = tf.add(output, 1)
		output = tf.multiply(output, 0.5)

	if COST == "MSE":

		# Mean Squared Estimate - the simplist cost function (MSE)

		cost = tf.reduce_mean(tf.square(Y - output)) 
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

	else:

		# Average Cross Entropy - better behaviour and learning rate	

		cost = - tf.reduce_mean( (y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output)  )
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


	init = tf.initialize_all_variables()
	sess = tf.Session()
	saver = tf.train.Saver()  
	

	with tf.Session() as sess:
		
		sess.run(init)
		
		if isTrain == 'Y' or isTrain == 'y':
		
			if isTrain == 'y':
				
				saver.restore(sess, './model.ckpt')
		
			for i in range(N_STEPS):
			
				sess.run(train_step, feed_dict={x_: X, y_: Y})
				
				if i % N_EPOCH == 0 and i > 0:
				
					saver.save(sess, './model.ckpt')
					print('Batch ', i)
					print('Inference \n', sess.run(output, feed_dict={x_: X, y_: Y}))
					print('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))
					
		else:
			
			saver.restore(sess, './model.ckpt')
			np.savetxt('Output.csv', sess.run(output, feed_dict={x_: X,}), delimiter=",")

	
