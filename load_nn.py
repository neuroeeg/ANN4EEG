import math
import random
import time
import func
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import tensorflow as tf
from scipy import signal

cl_num=10 # количество классов при обучении
ch_num=4

sig=[]
y=[]
sig_len=[]
ch_num_list=[]
preps=['are','dia']

for k in preps:
    sig_0,y_0,sig_len_0,ch_num_0=func.get_batch('train_dataset_'+k+'_test_2000.txt')
    sig.append(sig_0)
    y.append(y_0)
    sig_len.append(sig_len_0)
    ch_num_list.append(ch_num_0)


learning_rate=0.00015

cl=[]
cl_car=[]

for i in range(len(y)):
    for j in y[i]:
        c=i
        cl.append([1 if j==c else 0 for j in range(cl_num)])


step=10
n=100
n_min=5

x_data = tf.placeholder(tf.float32, [None, ch_num, sig_len[0]])
#tr_ph_shaped = tf.reshape(tr_ph, [-1, sig_len , 1])
x_shaped = tf.reshape(x_data, [-1, sig_len[0] , ch_num])
x_data_sh = tf.reshape(x_shaped, [-1, sig_len[0]*ch_num])
nd_pl=tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, cl_num])
print('begin')

out_clas, logits_clas, logits_sim,  bn = func.simulator_classifier(x_shaped,  sig_len[0], n, nd_pl, ch_num, cl_num)
# out_clas, logits_clas = func.simulator_classifier(x_shaped,  sig_len, n, nd_pl, ch_num,  cl_num)
#y_s=tf.tanh(y_)

t_vars = tf.trainable_variables()
s_vars = [var for var in t_vars if (var.name.startswith("sim_clas") or var.name.startswith("sim"))]# or 
c_vars = [var for var in t_vars if (var.name.startswith("clas") or var.name.startswith("sim_clas"))]#)
#learning
loss_sim = tf.losses.mean_squared_error(x_data_sh, logits_sim)
# loss_sim = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_sim, labels = y)
accuracy_sim=tf.reduce_mean(loss_sim)
optimiser_sim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_sim,var_list=s_vars)

loss_clas = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_clas, labels = y)
optimiser_clas = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_clas, var_list=c_vars)
correct_prediction = tf.equal(tf.argmax(out_clas, 1), tf.argmax(y, 1))
accuracy_clas = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
train_cu=[]
with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver()

    print('restore')

    ckpt = tf.train.get_checkpoint_state("checkpoint_dir")
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.save(sess, "checkpoint_dir/model.ckpt")

    
    print('test')
    win_eeg=10
    nd_test=[]
    test=[]

    for g in range(len(y)):
        nd_test=[[1 for i in range(n)]  for k in range(len(sig[g]))]
        test.append(sess.run(out_clas, feed_dict={x_data:sig[g][0:500], nd_pl: nd_test[0:500]}))
        title=preps[g]
        tt=func.fig(win_eeg,test[g], title)
   
    
    print("\nEnd!")

