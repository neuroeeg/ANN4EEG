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
file_name='train_dataset_set_7_1000.txt'#_'+prep+'


learning_rate=0.0001
batch_size=100
epochs=60

sig,y,sig_len,ch_num=func.get_batch(file_name)
sigt,yt,sig_lent,ch_numt=func.get_batch('test_dataset_set_7_1000.txt')# read file with dataset
print(yt)

#list of drugs
pharm=['gab','dia','car','pre','esl','phe','are','cor','pic','pil','chl']#
cl_num=len(pharm)

osob_num=5
nn=osob_num*len(pharm)

cl=[]
for i in y:
    for j in pharm:
        if i==j:
            c=pharm.index(j)
            cl.append([1 if j==c else 0 for j in range(cl_num)]) 

clt=[]
for i in yt:
    for j in pharm:
        if i==j:
            c=pharm.index(j)
            clt.append([1 if j==c else 0 for j in range(cl_num)]) 
# print(y[0:10])
# print(cl[0:10])
# print(cl)
r=[i for i in range(len(cl))]
rt=[i for i in range(len(clt))]
random.shuffle(r)
random.shuffle(rt)
# print(r)


ra=[(random.random()+0.5) for i in range(len(cl))]
sig_n=[[[sig[i][k][j]*ra[i] for j in range(len(sig[i][k]))] for k in range(len(sig[i]))] for i in range(len(cl))]
sig_nt=[[[sigt[i][k][j] for j in range(len(sigt[i][k]))] for k in range(len(sigt[i]))] for i in range(len(clt))]

sig_r=[[[sig[i][k][j]*ra[i] for j in range(len(sig[i][k]))] for k in range(len(sig[i]))] for i in r]
cl_r=[cl[i] for i in r]

sig_rt=[[[sigt[i][k][j] for j in range(len(sigt[i][k]))] for k in range(len(sigt[i]))] for i in rt]
cl_rt=[clt[i] for i in rt]


step=5
n=100
n_min=1
x_data = tf.placeholder(tf.float32, [None, ch_num, sig_len])
#tr_ph_shaped = tf.reshape(tr_ph, [-1, sig_len , 1])
x_shaped = tf.reshape(x_data, [-1, sig_len , ch_num])
x_data_sh = tf.reshape(x_shaped, [-1, sig_len*ch_num])
nd_pl=tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, cl_num])
print('begin')

out_clas, logits_clas, logits_sim,  bn = func.simulator_classifier(x_shaped,  sig_len, n, nd_pl, ch_num,  cl_num)

t_vars = tf.trainable_variables()
s_vars = [var for var in t_vars if (var.name.startswith("sim_clas") or var.name.startswith("sim"))]# or 
c_vars = [var for var in t_vars if (var.name.startswith("clas") or var.name.startswith("sim_clas"))]#)
#learning
loss_sim = tf.losses.mean_squared_error(x_data_sh, logits_sim)
accuracy_sim=tf.reduce_mean(loss_sim)
optimiser_sim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_sim,var_list=s_vars)

loss_clas = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_clas, labels = y)
optimiser_clas = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_clas, var_list=c_vars)
correct_prediction = tf.equal(tf.argmax(out_clas, 1), tf.argmax(y, 1))
accuracy_clas = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
train_cu=[]
train_cu_s=[]
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
        
        for i in range(int(total_batch-1)):
            batch = sig_r[(i*batch_size):((i+1)*batch_size)]
            batch_y=cl_r[(i*batch_size):((i+1)*batch_size)]
            batch_nd=nd[(i*batch_size):((i+1)*batch_size)]
            # _= sess.run(optimiser_sim, feed_dict={x_data:batch, nd_pl: batch_nd, y:batch_y})
            _= sess.run(optimiser_clas, feed_dict={x_data:batch, nd_pl: batch_nd, y:batch_y})
        yo_c=sess.run([accuracy_clas], feed_dict={x_data:sig_rt[0:500], nd_pl: nd_test[0:500], y: cl_rt[0:500]})
        yo_s=sess.run([accuracy_sim], feed_dict={x_data:sig_rt[0:500], nd_pl: nd_test[0:500], y: cl_rt[0:500]})
        yo_c_2=sess.run([accuracy_clas], feed_dict={x_data:sig_r[0:500], nd_pl: nd_test[0:500], y: cl_r[0:500]})
        pr=[]
        mn=len(pharm)
        
        #probability of correct detection
        for ui in range(nn):
            yo_all=sess.run(out_clas, feed_dict={x_data:sig_nt[int(ui*len(sig_nt)/nn):int((ui+1)*len(sig_nt)/nn)], nd_pl: nd_test[int(ui*len(sig_nt)/nn):int((ui+1)*len(sig_nt)/nn)], y: clt[int(ui*len(sig_nt)/nn):int((ui+1)*len(sig_nt)/nn)]}).tolist()
            a,b =func.prob(10, yo_all, clt[int(ui*len(sig_nt)/nn):int((ui+1)*len(sig_nt)/nn)])
            if (ui>nn-7):
                print(a)
            print(b)
            pr.append(b)


        print('prod_test:'+str(sum(pr)/len(pr)))
        low=0
        for un in pharm:
            
            print(un+'_:'+str(sum(pr[low*5:(low+1)*5])/5))
            low=low+1
        print(yo_c)
        # print(yo_s)
        print(yo_c_2)
        win_eeg=5

        train_cu.append(sum(pr)/len(pr))
        # train_cu_s.append(yo_s)
    fig, ax = plt.subplots()
    ax.plot(train_cu)
    ax.grid()
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('probability of correct detection')
    plt.show()
    
    saver.save(sess, "checkpoint_dir/model.ckpt")


    clr=['ro','bo','go','yo','go','yo']#,'wo'
    avvr=10
    dist_list_total=[]
    for lol in range(avvr):
        mid_claster=[[] for i in range(cl_num)]
        for i in np.arange(int(lol*len(sig_r)/avvr),int((lol+1)*len(sig_r)/avvr),1):#
            print(i)
            h=sess.run(bn, feed_dict={x_data:[sig_r[i]], nd_pl: [nd_test[i]]}).tolist()
            for j in cl_r[i]:
                if j==1:
                    mid_claster[cl_r[i].index(max(cl_r[i]))].append(h[0][0:n])
        avr_claster=[[] for i in range(cl_num)]
        for j in range(cl_num):
            for k in range(n):
                avr_claster[j].append(sum([mid_claster[j][o][k] for o in range(len(mid_claster[j]))])/len(mid_claster[j]))
            
        print(avr_claster)
        print(len(avr_claster))
        dist_list=[]
        for c in range(cl_num):
            sum_sqr = 0
            distance=[]
            x=['1_gab','2_dia','3_car','4_pre','5_esl','6_phe','7_are','8_cor','9_pic','10_pil','11_chl']

            for r in range(cl_num):
                sum_sqr = 0
                for i, j in zip(avr_claster[c],avr_claster[r]):
                    sum_sqr += (i-j)**2
                distance.append(math.sqrt(sum_sqr))
            print(distance)
            dist_list.append(distance)
        dist_list_total.append(dist_list)

    for p in range(cl_num):

        sred_res=[sum([dist_list_total[i][p][j] for i in range(len(dist_list_total))])/len(dist_list_total) for j in range(len(dist_list_total[0][0]))]
        print(sred_res)
        res_sko=[(sum([(dist_list_total[i][p][j]-sred_res[j])*(dist_list_total[i][p][j]-sred_res[j]) for i in range(len(dist_list_total))])/len(dist_list_total))**0.5 for j in range(len(dist_list_total[0][0]))]
        res_max=[sred_res[j]+res_sko[j]/2 for j in range(len(dist_list_total[0][0]))]
        res_min=[sred_res[j]-res_sko[j]/2 for j in range(len(dist_list_total[0][0]))]

        plt.plot(x,sred_res)

        plt.fill_between(x, res_min, res_max, color = 'b', alpha = 0.1)
        plt.title(pharm[p])
        plt.show()


    y_list_c=[]
    k=0
    for t in range(n_min,n,step):
        print(t)
        numb=[t for i in range(len(sig_r))]
        nd=[[1 if i<numb[k] else 0 for i in range(n)] for k in range(len(sig_r))]
        yo_l=sess.run([accuracy_clas], feed_dict={x_data: sig_r[0:500],  nd_pl: nd[0:500], y:cl_r[0:500]})
        y_list_c.append(yo_l)
       
        k=k+1
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_min,n,step).tolist(),y_list_c)
    ax.grid()
    ax.set_xlabel('number of parameters')
    ax.set_ylabel('probability of correct detection')
    plt.show()
    y_list_s=[]
    k=0
    for t in range(n_min,n,step):
        
        numb=[t for i in range(len(sig_r))]
        nd=[[1 if i<numb[k] else 0 for i in range(n)] for k in range(len(sig_r))]
        yo_l=sess.run([accuracy_sim], feed_dict={x_data: sig_r[0:500],  nd_pl: nd[0:500], y:cl_r[0:500]})
        y_list_s.append(yo_l)
       
        k=k+1
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_min,n,step).tolist(),y_list_s)
    ax.grid()
    ax.set_xlabel('number of parameters')
    ax.set_ylabel('standard deviation')
    plt.show()

   
    print("\nEnd!")

