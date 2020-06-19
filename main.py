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
from mpl_toolkits.mplot3d import Axes3D
prep='kar'
file_name='train_dataset_ami_are_atr_caf_chl_dia_hal_cor_per_car_2500_500hz.txt'#_'+prep+'


learning_rate=0.0001
batch_size=200
epochs=20

sig,y,sig_len,ch_num=func.get_batch(file_name)

pharm=['ami','are','atr','caf','chl','dia','hal','cor','per','car']
cl_num=len(pharm)

cl=[]
for i in y:
    for j in pharm:
        if i==j:
            c=pharm.index(j)
            cl.append([1 if j==c else 0 for j in range(cl_num)]) 
# print(y[0:10])
# print(cl[0:10])

# print(cl)
r=[i for i in range(len(cl))]
random.shuffle(r)
# print(r)


ra=[(random.random()+0.5) for i in range(len(cl))]
print(ra[0:10])
print(len(sig))
print(len(sig[0]))
print(len(sig[0][0]))
print(sig[0][0][0])
sig_r=[[[sig[i][k][j]*ra[i] for j in range(len(sig[i][k]))] for k in range(len(sig[i]))] for i in r]#(random.random()*0.5+0.5)#
print(ra[0:10])
cl_r=[cl[i] for i in r]
sig=[]
# sig_f=[]
# y=0
# for i in sig_r:
#     y=y+1
#     print(y)
#     sig_filt=[]
#     for j in range(ch_num):
#         b, a = signal.butter(3, 100*2/2000 , btype='low')#[2*10/2000, 70*2/2000]
#         sig_filt.append(signal.filtfilt(b, a, i[j]))
#     sig_f.append(sig_filt)
# plt.plot(sig_f[0][0])
# plt.show()


# sig_learn=sig_r[0:int(4/5*len(sig_r))]#]#int(2/3*len(sig_r))
# sig_test=sig_r[int(4/5*len(sig_r)):len(sig_r)]#



# # print(sig_learn)

# cl_learn=cl_r[0:int(4/5*len(cl_r))]#[0:i]#int(2/3*len(cl_r))
# cl_test=cl_r[int(4/5*len(cl_r)):len(cl_r)]


step=10
n=100
n_min=5

x_data = tf.placeholder(tf.float32, [None, ch_num, sig_len])
#tr_ph_shaped = tf.reshape(tr_ph, [-1, sig_len , 1])
x_shaped = tf.reshape(x_data, [-1, sig_len , ch_num])
x_data_sh = tf.reshape(x_shaped, [-1, sig_len*ch_num])
nd_pl=tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, cl_num])
print('begin')

out_sim, logits_clas, logits_sim,  bn = func.simulator_classifier(x_shaped,  sig_len, n, nd_pl, ch_num,  cl_num)
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
correct_prediction = tf.equal(tf.argmax(out_sim, 1), tf.argmax(y, 1))
accuracy_clas = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
train_cu=[]
with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver()
    # saver.save(sess, "checkpoint_dir/model.ckpt")
    total_size=len(sig_r)
    print('start')
    total_batch=int(total_size/batch_size)
    print(total_batch)
    numb=[np.random.randint(n_min, n) for i in range(total_size)]
    nd=[[1 if i<numb[k] else 0 for i in range(n)] for k in range(total_size)]
    nd_test=[[1 for i in range(n)]  for k in range(len(sig_r))]
    for epoch in range(epochs):
        print(epoch)
        random.shuffle(nd)
        
        for i in range(int(total_batch-10)):
            batch = sig_r[(i*batch_size):((i+1)*batch_size)]
            batch_y=cl_r[(i*batch_size):((i+1)*batch_size)]
            batch_nd=nd[(i*batch_size):((i+1)*batch_size)]
            _= sess.run(optimiser_sim, feed_dict={x_data:batch, nd_pl: batch_nd, y:batch_y})
            _= sess.run(optimiser_clas, feed_dict={x_data:batch, nd_pl: batch_nd, y:batch_y})
        # yo_s=sess.run([accuracy_sim], feed_dict={x_data:sig_test[0:500], nd_pl: nd_test[0:500]})
        yo_c=sess.run([accuracy_clas], feed_dict={x_data:sig_r[len(sig_r)-100:len(sig_r)], nd_pl: nd_test[len(sig_r)-100:len(sig_r)], y: cl_r[len(sig_r)-100:len(sig_r)]})
        yo_s=sess.run([accuracy_sim], feed_dict={x_data:sig_r[len(sig_r)-100:len(sig_r)], nd_pl: nd_test[len(sig_r)-100:len(sig_r)], y: cl_r[len(sig_r)-100:len(sig_r)]})
        yo_c_2=sess.run([accuracy_clas], feed_dict={x_data:sig_r[0:500], nd_pl: nd_test[0:500], y: cl_r[0:500]})
        print(yo_c)
        print(yo_s)
        print(yo_c_2)
        win_eeg=5

        train_cu.append(yo_c[0])

        
        
    saver.save(sess, "checkpoint_dir/model.ckpt")
    plt.plot(train_cu)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    clr=['ro','bo','go','yo','mo','ko','co','mo','ko','co']#,'wo'
    mid_claster=[[] for i in range(cl_num)]
    for i in np.arange(len(sig_r)-200,len(sig_r),1):
        print(i)
        h=sess.run(bn, feed_dict={x_data:[sig_r[i]], nd_pl: [nd_test[i]]}).tolist()
        for j in cl_r[i]:
            if j==1:
                mid_claster[cl_r[i].index(max(cl_r[i]))].append(h[0][0:n_min])
        plt.plot(h[0][0],h[0][1],clr[sig_r[i].index(max(sig_r[i]))])
    plt.show()
    # print(mid_claster)
    # print(mid_claster[0])
    # print(mid_claster[0][0])

    avr_claster=[[] for i in range(cl_num)]
    for j in range(cl_num):
        for k in range(n_min):
            avr_claster[j].append(sum([mid_claster[j][o][k] for o in range(len(mid_claster[j]))])/len(mid_claster[j]))
    print(avr_claster)
    print(len(avr_claster))

    for c in range(cl_num):
        sum_sqr = 0
        distance=[]
        x=['1_ami','2_are','3_atr','4_caf','5_chl','6_dia','7_hal','8_cor','9_per','10_car']
        for r in range(cl_num):
            sum_sqr = 0
            for i, j in zip(avr_claster[c],avr_claster[r]):
                sum_sqr += (i-j)**2
            distance.append(math.sqrt(sum_sqr))
        print(distance)
        plt.title(pharm[c])
        plt.plot(x,distance)
        plt.show()

    # h_t=[]
    # for j in range(len(h[0])):
    #     h_trans=0
    #     for i in range(len(h)):
    #         h_trans+=abs(h[i][j])
    #     h_t.append(h_trans)
    # plt.plot(h_t)
    # plt.show()

    y_list_c=[]
    k=0
    for t in range(n_min,n,step):
        print(t)
        numb=[t for i in range(len(sig_r))]
        nd=[[1 if i<numb[k] else 0 for i in range(n)] for k in range(len(sig_r))]
        yo_l=sess.run([accuracy_sim], feed_dict={x_data: sig_r[0:500],  nd_pl: nd[0:500], y:cl_r[0:500]})
        y_list_c.append(yo_l)
       
        k=k+1
    plt.plot(np.arange(n_min,n,step).tolist(),y_list_c)
    plt.show()
    y_list_s=[]
    k=0
    for t in range(n_min,n,step):
        
        numb=[t for i in range(len(sig_r))]
        nd=[[1 if i<numb[k] else 0 for i in range(n)] for k in range(len(sig_r))]
        yo_l=sess.run([accuracy_sim], feed_dict={x_data: sig_r[0:100],  nd_pl: nd[0:100], y:cl_r[0:100]})
        y_list_s.append(yo_l)
       
        k=k+1
    plt.plot(np.arange(n_min,n,step).tolist(),y_list_s)
    plt.show()
    plt.plot(y_list_c,y_list_s)
    plt.show()
   
    print("\nEnd!")

