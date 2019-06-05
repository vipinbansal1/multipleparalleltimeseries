
import tensorflow as tf
import numpy as np
import pandas as pd


date_data = pd.read_csv("date_data.csv")
date_data = date_data.drop(columns=["Date"])

def dateBatch(sequences, n_steps):
    X = list()
    y = list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        seq_x, seq_y = sequences.values[i:end_ix, :], sequences.values[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
    

tf.reset_default_graph()
n_steps = 5
n_inputs = 3
n_neurons = 100
n_outputs = 3
learning_rate = 0.01
epoch = 10000


X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_outputs])

basicCell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation=tf.nn.relu)

outputs, states = tf.nn.dynamic_rnn(basicCell, X, dtype=tf.float32)
logits = tf.layers.dense(states, n_outputs)

loss = tf.reduce_mean(tf.square(logits-y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for iteration in range(epoch):
        X_batch, y_batch = dateBatch(date_data, n_steps)
        op = sess.run(training_op, feed_dict={X:X_batch, y :y_batch})
        if(iteration%100 == 0):
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(mse)
    unseenSample = logits.eval(feed_dict={X:np.array([[[20,6,2019],[21,6 ,2019],
                                                       [22,6,2019],[23,6,2019],
                                                       [24,6,2019]]])})
    print(np.ceil(unseenSample))
    
