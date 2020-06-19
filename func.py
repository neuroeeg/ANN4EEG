import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
import time
import sounddevice as sd
import random
import math
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import os
from numpy.fft import *
import copy

def make_batch(file_name):
    fi_num=0
    line=[]
    total_len_f=sum([1 for line in open(file_name,'r')])
    print('line num:'+ str(total_len_f))
    set_len = 4000000#total_len_f#

    with open(file_name,'r') as file:
        
        for line_for_cnt in file:
            if (fi_num%10000==0):
                print('line read:'+str(fi_num/total_len_f*100)+'%')
            fi_num=fi_num+1
            if(fi_num>1400000 and fi_num<set_len):
                line_for_cnt=line_for_cnt.replace('\n','')
                line_for_cnt=line_for_cnt.split(',')
                line_for_cnt=[float(line_for_cnt[i]) for i in range(len(line_for_cnt))]
                line.append([line_for_cnt[14]/100, line_for_cnt[16]/100, line_for_cnt[19]/100, line_for_cnt[21]/100])# 0,0,0,0#line_for_cnt[14], line_for_cnt[16], line_for_cnt[19], line_for_cnt[21]
                
    return line

def get_batch(file_name):
    fi_num=0
    line=[]
    total_len_f=sum([1 for line in open(file_name,'r')])
    print('line num:'+ str(total_len_f))
    set_len = total_len_f#

    with open(file_name,'r') as file:  
        
        sig=[]
        y=[]
        for line_for_cnt in file:
            x=[]
            if (fi_num%100==0):
                print('line read:'+str(fi_num/total_len_f*100)+'%')
            fi_num=fi_num+1
            if(fi_num==1):
                ch_num=sum(1 for i in line_for_cnt if i=="]")
            line_for_cnt=line_for_cnt.replace('\n','')
            line_for_cnt=line_for_cnt.replace('][',', ')
            line_for_cnt=line_for_cnt.replace(']','')
            line_for_cnt=line_for_cnt.replace('[','')
            line_for_cnt=line_for_cnt.split(', ')
            # print(line_for_cnt)
            sig_len=int((len(line_for_cnt)-1)/ch_num)
            # print(sig_len)
            for i in range(ch_num):
                x.append([float(k) for k in line_for_cnt[i*sig_len:(i+1)*sig_len]])
            sig.append(x)
            y.append(line_for_cnt[len(line_for_cnt)-1])
                
    return sig,y,sig_len,ch_num


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_h, filter_w, pool_shape, name):
    
    conv_filt_shape = [filter_h, filter_w, num_input_channels,
                      num_filters]

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    # weights = tf.get_variable("W", shape=conv_filt_shape, initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    # bias = tf.get_variable("b", shape=[num_filters], initializer=tf.contrib.layers.xavier_initializer())
    out_layer = tf.nn.conv2d(input_data, weights, strides=[1, 1, 1, 1], padding='SAME') 
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    ksize = pool_shape
    window_shape=[]
    window_shape.append(pool_shape)
    strides = []
    strides.append(ksize)
    out_layer = tf.nn.pool(out_layer, [pool_shape,pool_shape], pooling_type="MAX",padding='SAME',strides=[pool_shape,pool_shape])

    return out_layer, weights

def classifier_conv(input_data,  sig_len, ch_num, cl_num, reuse=False):  
    with tf.variable_scope('classifier', reuse=reuse):
        win_len=int(sig_len/20)
        n=15
        x_shaped = tf.reshape(input_data, [-1, sig_len, ch_num , 1])
        layer1, weights = create_new_conv_layer(x_shaped, 1, n, 100, 4, 4, name='layer0')
        # layer2, weights = create_new_conv_layer(layer1, n, int(n/2), 20, 4, 1, name='layer2')
        # layer3, weights = create_new_conv_layer(layer2, n, n, 10, 4, 1, name='layer3')
        flattened = tf.reshape(layer1, [-1, int(sig_len*n*ch_num/4/4)])
        flattened=tf.nn.dropout(flattened,0.5)
        dense_layer1 = tf.layers.dense(flattened, cl_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        y_ = tf.nn.softmax(dense_layer1)
        # flattened = tf.reshape(input_data, [-1, sig_len*ch_num])
        # h1 = tf.layers.dense(flattened, sig_len, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # h1 = tf.nn.relu(h1)
        # # h1=tf.nn.dropout(h1,0.5)
        # # h3t = tf.layers.dense(h2t, h_size*2 , activation=None)
        # #h3t = tf.nn.relu(h3t)
        # dense_layer1 = tf.layers.dense(h1, cl_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # # logits_shape=tf.reshape(logits,[-1,sig_len])
        # #out=tf.sign(logits_shape)*2
        # y_=tf.nn.softmax(dense_layer1)# tf.nn.relu(logits_shape)#  tf.sin(logits)#

        return dense_layer1, y_

def simulator(input_data,  sig_len, n, nd_pl, ch_num,  reuse=False):  
    #sig_len = len(input_data[0])
    h_size=sig_len
    
    with tf.variable_scope('simulator', reuse=reuse):
        # h1t = tf.layers.dense(input_data, h_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#
        # h1t = tf.nn.relu(h1t)# tf.nn.leaky_relu(h1t,alpha=0.2)#
        #nd_1=tf.layers.dense(h1t, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        #h2t = tf.layers.dense(h1t, 20, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        #h2t = tf.nn.tanh(h2t)# tf.nn.leaky_relu(h1t,alpha=0.2)#
        #nd=np.random.randint(10,20)
        #print('np=', str(n))
        h2t = tf.layers.dense(input_data, n, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        #print(int(nd))
        h2t = tf.nn.relu(h2t)*nd_pl# tf.nn.leaky_relu(h3t,alpha=0.2)
        # h3t = tf.layers.dense(h2t, h_size , activation=None)
        # h3t = tf.nn.relu(h3t)
        logits = tf.layers.dense(h2t, sig_len*ch_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        logits_shape=tf.reshape(logits,[-1,sig_len*ch_num])
        #out=tf.sign(logits_shape)*2
        out=tf.nn.softmax(logits_shape)# tf.nn.relu(logits_shape)#  tf.sin(logits)#
        #out_shape=tf.reshape(out,[-1,num_ch,sig_len])
        return out, logits_shape, h2t
    '''with tf.variable_scope('sigmator', reuse=reuse):
        win_len=10
        layer1_s = create_new_conv_layer(input_data, 1, 10, win_len, 10, name='layer1')

        flattened_s = tf.reshape(layer1_s, [-1, int(10*sig_len/10)])

        wd1_s = tf.Variable(tf.truncated_normal([int(10*sig_len/10), sigma_len], stddev=0.01))
        bd1_s = tf.Variable(tf.truncated_normal([sigma_len], stddev=0.01))
        dense_layer1_s = tf.matmul(flattened_s, wd1_s) + bd1_s
        y_s = tf.nn.relu(dense_layer1_s)
        return y_s, dense_layer1_s'''

def simulator_classifier(input_data,  sig_len, n, nd_pl, ch_num,  cl_num, reuse=False):  
    #sig_len = len(input_data[0])
    h_size=sig_len
    with tf.variable_scope('sim_clas', reuse=reuse):
        m=10
        x_shaped = tf.reshape(input_data, [-1, sig_len, ch_num,1])
        layer1, weights = create_new_conv_layer(x_shaped, 1, m, 100, 4, 1, name='layer1')
        # layer2, weights1 = create_new_conv_layer(layer1, m, m, 20, 4, 4, name='layer2')
        # # layer3, weights = create_new_conv_layer(layer2, n, n, 10, 4, 1, name='layer3')
        # flattened = tf.reshape(layer2, [-1, int(sig_len*m*ch_num/4/4)])
        flattened = tf.reshape(layer1, [-1, int(sig_len*m*ch_num)])
        flattened=tf.nn.dropout(flattened,0.5)
        h2t = tf.layers.dense(flattened, n, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # x_shaped = tf.reshape(input_data, [-1, sig_len*ch_num])
        # h1t = tf.layers.dense(x_shaped, 100, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#
        # h1t = tf.nn.relu(h1t)# tf.nn.leaky_relu(h1t,alpha=0.2)#
        # # h1t = tf.layers.dense(h1t, 50, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#
        # # h1t = tf.nn.relu(h1t)# tf.nn.leaky_relu(h1t,alpha=0.2)#
        # h2t = tf.layers.dense(h1t, n, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())#
        # # # h2t = tf.nn.relu(h2t)
        # # # bn = tf.nn.relu(h2t)*nd_pl
        bn = (h2t)*nd_pl

    with tf.variable_scope('sim', reuse=reuse):
    #     logits_sim = tf.layers.dense(bn, sig_len*ch_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    #     logits_shape_sim=tf.reshape(logits_sim,[-1,sig_len*ch_num])
    #     out_sim=tf.nn.tanh(logits_shape_sim)*5
        h1c=tf.layers.dense(bn, 500, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        h1c=tf.nn.relu(h1c)

        logits_sim = tf.layers.dense(h1c, sig_len*ch_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('clas', reuse=reuse):
        # win_len=int(sig_len/20)
        # m=15
        # x_shaped = tf.reshape(input_data, [-1, sig_len, ch_num , 1])
        # layer1, weights = create_new_conv_layer(x_shaped, 1, m, 200, 3, 1, name='layer1')
        # # layer2, weights = create_new_conv_layer(layer1, n, int(n/2), 20, 4, 1, name='layer2')
        # # layer3, weights = create_new_conv_layer(layer2, n, n, 10, 4, 1, name='layer3')
        # flattened = tf.reshape(layer1, [-1, int(sig_len*m*ch_num/1)])
        # flattened=tf.nn.dropout(flattened,0.5)
        # logits_clas = tf.layers.dense(bn, 40, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # logits_clas = tf.nn.relu(logits_clas)

        
        # flattened = tf.reshape(logits_clas, [-1, sig_len*ch_num])
        # out_clas = flattened
        logits_clas = tf.layers.dense(bn, cl_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out_clas=tf.nn.softmax(logits_clas)#
        return  out_clas, logits_clas, logits_sim, bn
        # return out_clas, logits_clas

def prob(win, test_prob, cl):
    # win=10
    total_len=int(len(test_prob)/win)
    probab=0
    for i in range(total_len):
        test_pr=[]
        tr_test=np.transpose(test_prob[win*i:(i+1)*win])
        for pr in range(len(tr_test)):
            test_pr.append(sum(tr_test[pr]))
        # if cl[0].index(max(cl[0]))==0:
            # print(test_pr)
        if (test_pr.index(max(test_pr))==cl[0].index(max(cl[0]))):
            probab=probab+1
    return round(probab/total_len, 2)


def fig(win, test_prob, title):
    # win=10
    total_len=int(len(test_prob)/win)
    probab=0
    tot_list=[]
    for i in range(total_len):
        test_pr=[]
        tr_test=np.transpose(test_prob[win*i:(i+1)*win])
        for pr in range(len(tr_test)):
            test_pr.append(sum(tr_test[pr])/win)
        tot_list.append(test_pr)


    sred_res=[sum([tot_list[i][j] for i in range(len(tot_list))])/len(tot_list) for j in range(len(tot_list[0]))]
    # x=[i for i in range(len(tot_list[0]))]
    x=['1_car','2_chl','3_ami','4_gab','5_caf','6_atr']
    # x=['car','chl','ami','gab','caf','atr']
    res_sko=[(sum([(tot_list[i][j]-sred_res[j])*(tot_list[i][j]-sred_res[j]) for i in range(len(tot_list))])/len(tot_list))**0.5 for j in range(len(tot_list[0]))]
    res_max=[sred_res[j]+res_sko[j]/2 for j in range(len(tot_list[0]))]
    res_min=[sred_res[j]-res_sko[j]/2 for j in range(len(tot_list[0]))]

    plt.plot(x,sred_res)

    plt.fill_between(x, res_min, res_max, color = 'b', alpha = 0.1)
    plt.title(title)
    plt.show()

    return sred_res
