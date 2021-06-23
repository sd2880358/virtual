from libsvm.svmutil import *
import pandas as pd

y, x = svm_read_problem('./australian_scale')
features = convert_to_list(x)
data = pd.DataFrame(features, dtype='float32')
data.to_csv("./c_australian_scale", index=False)


def covert_to_list(data):
    l = []
    for i in range(len(data)):
        tmp = []
        for j in range(len(data[i])):
            tmp.append(data[i][j])
        l.append(tmp)
    return l