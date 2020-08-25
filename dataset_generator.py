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


#dlina obrezannogo signala dlia dataseta
sig_len = 1000
ch_num = 4

for fold in range(5):

    for i in files_for_set:
        file_name=i
        print(file_name[0:3])
        #funkcia schitivania iz ASCII faila
        l=func.make_batch(file_name)
        print('read complite')
        #virezanie neghnogo kanakla
        x=[]
        for i in range(ch_num):
            x.append([l[j][i] for j in range(len(l))])

        # #grafik
        # for i in range(ch_num):
        #     plt.plot(x[i])
        # plt.show()

        #dozapis v fail dataseta (postrochno)
        f = open('train_dataset_set_7_1000_fold_'+str(fold)+'.txt', 'a')
        total_size=int(len(x[0])/sig_len)
        l1=int(fold*total_size/5)
        l2=int((fold+1)*total_size/5)
        for i in np.arange(0,l1,1):#total_size
            print('set_len write:'+str(i))
            for j in range(ch_num):
                # print(len(x))
                # print(len(x[0]))
                f.write(str(x[j][i*sig_len:(i+1)*sig_len]))
            f.write(', '+file_name[0:3]+'\n')

        for i in np.arange(l2,total_size,1):#total_size
            print('set_len write:'+str(i))
            for j in range(ch_num):
                # print(len(x))
                # print(len(x[0]))
                f.write(str(x[j][i*sig_len:(i+1)*sig_len]))
            f.write(', '+file_name[0:3]+'\n')
        f.close()

        ft= open('test_dataset_set_7_1000_fold_'+str(fold)+'.txt', 'a')
        for i in np.arange(l1,l2,1):#total_size
            print('set_len write:'+str(i))
            for j in range(ch_num):
                # print(len(x))
                # print(len(x[0]))
                ft.write(str(x[j][i*sig_len:(i+1)*sig_len]))
            ft.write(', '+file_name[0:3]+'\n')
        ft.close()
    print('eeeeeend!!')
