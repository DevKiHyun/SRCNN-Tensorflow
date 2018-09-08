import glob
import numpy as np
import scipy.io
import os

train_input_path = './train_91_input/{}'
train_label_path = './train_91_label/{}'
train_path = [train_input_path, train_label_path]
test_set = ['Set5']
test_path = []
for elem in test_set:
    y_ch_path = './' + elem + '/y_ch/{}'
    test_path.append(y_ch_path)
    y_ch_2x_path = './' + elem + '/y_ch_2x/{}'
    test_path.append(y_ch_2x_path)
    y_ch_3x_path = './' + elem + '/y_ch_3x/{}'
    test_path.append(y_ch_3x_path)
    y_ch_4x_path = './' + elem + '/y_ch_4x/{}'
    test_path.append(y_ch_4x_path)
    color_path = './'+elem+'/color/{}'
    test_path.append(color_path)

for path in train_path:
    list = glob.glob(path.format('*.mat'))
    for file in list:
        print(file, ' --> npy')
        filename = os.path.basename(file)[:-4]
        mat = scipy.io.loadmat(file)['patch']
        np.save('{}.npy'.format(path.format(filename)), mat)
        os.remove(file)

for path in test_path:
    list = glob.glob(path.format('*.mat'))
    for file in list:
        print(file, '--> npy')
        filename = os.path.basename(file)[:-4]
        mat = scipy.io.loadmat(file)['img']
        np.save('{}.npy'.format(path.format(filename)), mat)
        os.remove(file)
