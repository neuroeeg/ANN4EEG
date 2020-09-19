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

# создание выборки для формирования датасета
def make_batch(file_name):
    fi_num=0
    line=[]
    total_len_f=sum([1 for line in open(file_name,'r')])
    print('line num:'+ str(total_len_f))
    set_len = total_len_f#
    # открываем файл формата "qwe_nn_data.txt"
    with open(file_name,'r') as file:
        
        for line_for_cnt in file:
            # пропускаем начало (в последней версии это необязательно)
            if (fi_num%10000==0):
                #выводим информацию о том сколько процентов файла считано
                print('line read:'+str(fi_num/total_len_f*100)+'%')
            fi_num=fi_num+1
            if(fi_num>10 and fi_num<set_len):#
                line_for_cnt=line_for_cnt.replace('\n','')
                line_for_cnt=line_for_cnt.split(',')
                line_for_cnt=[float(line_for_cnt[i]) for i in range(len(line_for_cnt))]
                # уменьшаем амплитуду, чтобы она стала порядка 1 (возможно стоит использовать нормировку)
                line.append([line_for_cnt[1]/100, line_for_cnt[2]/100, line_for_cnt[3]/100, line_for_cnt[4]/100])# 0,0,0,0#line_for_cnt[14], line_for_cnt[16], line_for_cnt[19], line_for_cnt[21]
                
    return line


#считывание данных из датасета в списки
def get_batch(file_name):
    fi_num=0
    line=[]
    total_len_f=sum([1 for line in open(file_name,'r')])
    print('line num:'+ str(total_len_f))
    set_len = total_len_f#
    #открываем файл датасета
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
            
    return sig,y,sig_len,ch_num# sig список содержащий отсчеты сигналов для каждого канала, у - метки классов для сигналов, sig_len - длина сигналов, ch_num - количество каналов

# сверточный слой ввиде функции
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

#архитектура нейронной сети
def simulator_classifier(input_data,  sig_len, n, nd_pl, ch_num,  cl_num, reuse=False):  
    #sig_len = len(input_data[0])
    h_size=sig_len
    with tf.variable_scope('sim_clas', reuse=reuse):
        m=10
        x_shaped = tf.reshape(input_data, [-1, sig_len, ch_num,1])
        layer1, weights = create_new_conv_layer(x_shaped, 1, m, 50, 4, 1, name='layer1')
        layer2, weights1 = create_new_conv_layer(layer1, m, m, 20, 4, 4, name='layer2')
        flattened = tf.reshape(layer2, [-1, int(sig_len*m*ch_num/16)])
        h2t = tf.layers.dense(flattened, n, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # бутылочное горлышко
        bn = (h2t)*nd_pl

    with tf.variable_scope('sim', reuse=reuse):
        h1c=tf.layers.dense(bn, 100, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        h1c=tf.nn.relu(h1c)

        logits_sim = tf.layers.dense(h1c, sig_len*ch_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('clas', reuse=reuse):

        logits_clas = tf.layers.dense(bn, cl_num, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out_clas=tf.nn.softmax(logits_clas)#
        return  out_clas, logits_clas, logits_sim, bn

def prob(win, test_prob, cl):
    # win=10
    total_len=int(len(test_prob)/win)
    probab=0
    test_all=[]
    for i in range(total_len):
        test_pr=[]
        tr_test=np.transpose(test_prob[win*i:(i+1)*win])
        for pr in range(len(tr_test)):
            test_pr.append(sum(tr_test[pr]))
        if (test_pr.index(max(test_pr))==cl[0].index(max(cl[0]))):
            probab=probab+1
    tr_test_all=np.transpose(test_prob)
    for pr in range(len(tr_test_all)):
        test_all.append(sum(tr_test_all[pr])) 
    return test_all, round(probab/total_len, 2)

#рисунки с СКО
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
    x=['1_ami','2_are','3_atr','4_caf','5_chl','6_dia','7_hal','8_cor','9_per','10_car']
    # x=['car','chl','ami','gab','caf','atr']
    res_sko=[(sum([(tot_list[i][j]-sred_res[j])*(tot_list[i][j]-sred_res[j]) for i in range(len(tot_list))])/len(tot_list))**0.5 for j in range(len(tot_list[0]))]
    res_max=[sred_res[j]+res_sko[j]/2 for j in range(len(tot_list[0]))]
    res_min=[sred_res[j]-res_sko[j]/2 for j in range(len(tot_list[0]))]

    plt.plot(x,sred_res)

    plt.fill_between(x, res_min, res_max, color = 'b', alpha = 0.1)
    plt.title(title)
    plt.show()

    return sred_res
