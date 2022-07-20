import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('model_data/yolobody0718.h5', compile=False)
weights = model.get_weights()

f = open('yolo_weights.h', 'w')
f.write('\n#include "defines.h"\n')
for index in range(len(weights)):
    w = weights[index]
    dw = np.array(w).ndim
    if dw == 1:
        f.write('\n static const DTYPE w%d[%d]={' % (index, len(w)))
        for iw in range(len(w)):
            f.write('%.32f, ' % w[iw])
        f.write('};\n')
    else:
        if dw == 4:
            sw = np.array(w).shape
            f.write('\n static const DTYPE w%d[%d][%d][%d][%d]={' % (index, sw[0], sw[1], sw[2], sw[3]))
            for iw in range(sw[0]):
                for jw in range(sw[1]):
                    for kw in range(sw[2]):
                        for lw in range(sw[3]):
                            f.write('%.32f, ' % w[iw][jw][kw][lw])
            f.write('};\n')

f.write('\n')
f.close()
'''
for index_cout in range(cout):
    for index_n in range(512):
        for index_h in range(1):
            for index_w in range(1):
                f.write('%.32f\n'%weight[(index_h*1+index_w)*512+index_n][index_cout])
'''
