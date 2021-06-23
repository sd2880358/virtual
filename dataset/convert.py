from libsvm.svmutil import *
import pandas as pd

def convert_to_csv(data, labels, dic):
    assert(len(data) == len(labels))
    for i in range(len(data)):
        data[i]['labels'] = labels[i]
    df = pd.DataFrame(data, dtype='float32')
    df.to_csv('./{}'.format(dic), index=False)


y, x = svm_read_problem('./australian_scale')
features = convert_to_csv(x, y, 'c_australian_scale')



