import numpy as np
import matplotlib.pyplot as plt
import func
import os

files = os.listdir()
files_for_set=[]
for i in files:
    if (i.endswith('.txt')==1):
        files_for_set.append(i)
print(files_for_set[0][0:3])


#lengh of signal
sig_len = 1000
ch_num = 4


for i in files_for_set:
    file_name=i
    print(file_name[0:3])
    #func for read ASCII 
    l=func.make_batch(file_name)
    print('read complite')
 
    x=[]
    for i in range(ch_num):
        x.append([l[j][i] for j in range(len(l))])

    # #figure
    # for i in range(ch_num):
    #     plt.plot(x[i])
    # plt.show()

    #write train set
    f = open('train_dataset_set_1000.txt', 'a')
    total_size=int(len(x[0])/sig_len)
    for i in np.arange(0,int(total_size-total_size/5),1):#total_size
        print('set_len write:'+str(i))
        for j in range(ch_num):
            f.write(str(x[j][i*sig_len:(i+1)*sig_len]))
        f.write(', '+file_name[0:3]+'\n')
    f.close()
    
    #write test set
    ft= open('test_dataset_set_1000.txt', 'a')
    for i in np.arange(int(total_size-total_size/5),int(total_size),1):#total_size
        print('set_len write:'+str(i))
        for j in range(ch_num):
            ft.write(str(x[j][i*sig_len:(i+1)*sig_len]))
        ft.write(', '+file_name[0:3]+'\n')
    ft.close()
print('end!')
