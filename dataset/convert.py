from libsvm.svmutil import *
import pandas as pd

def convert_to_list(data):
    l = []
    for i in range(len(data)):
        tmp = []
        for key in data[i].keys:
            tmp.append(data[i][key])
        l.append(tmp)
    return l


y, x = svm_read_problem('./australian_scale')
features = convert_to_list(x)
data = pd.DataFrame(features, dtype='float32')
data.to_csv("./c_australian_scale", index=False)


