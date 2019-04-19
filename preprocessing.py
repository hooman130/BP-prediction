import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#reading .mat data for all subjects
for s in [1,3,6,7,11,13,14,15]:
    print(s)
    #reading data from all 4 sensors
    file = sio.loadmat("Subject%d/data" % s)
    data1 = file['bioz1']
    data2 = file['bioz2']
    data3 = file['bioz3']
    data4 = file['bioz4']
    time_data = file['time_data']
    labels = file['raw_labels']
	
    series_len = 150 #length of the heartbeat sequences
    data = np.zeros((1,series_len,4))
    new_labels = pd.DataFrame()
    list_data = list()
	#extracting the heartbeats using the sample numbers provided along with the labels
    for i in range(labels.shape[0]):
        l = int(labels[i,1]) - int(labels[i,0]) + 1 #length of the beat
        d1 = data1[(int(labels[i,0])-1):int(labels[i,1])]
        d2 = data2[(int(labels[i,0])-1):int(labels[i,1])]
        d3 = data3[(int(labels[i,0])-1):int(labels[i,1])]
        d4 = data4[(int(labels[i,0])-1):int(labels[i,1])]
        t_dv = time_data[(int(labels[i,0])-1):int(labels[i,1])] # timing data
        t_dv = t_dv - t_dv[0]
        list_data.append(np.concatenate((d1,d2,d3,d4,t_dv), axis=1))
        new_labels = new_labels.append(pd.DataFrame({'SBP':[labels[i,3]],'DBP':[labels[i,2]]}))

    # saving data and labels into pickle format files
    f = open('Subject%d/list_data_time_100.pckl' % s, 'wb')
    pickle.dump(list_data, f)
    f.close()

    f = open('Subject%d/labels_100.pckl' % s, 'wb')
    pickle.dump(new_labels, f)
    f.close()