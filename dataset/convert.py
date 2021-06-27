from libsvm.svmutil import *
import pandas as pd

def convert_to_csv(data, labels, dic):
    assert(len(data) == len(labels))
    for i in range(len(data)):
        data[i]['labels'] = labels[i]
    df = pd.DataFrame(data, dtype='float32')
    df.to_csv('./{}'.format(dic), index=False)


files = ['australian', 'breast-cancer', 'diabetes', 'german.numer', 'svmguide2', 'svmguide4']
y, x = svm_read_problem('./australian_scale')
for file in files:
    y, x = svm_read_problem(file)
    convert_to_csv(x, y, 'c_{}'.format(file))



features = {'Sample code number': 'id number', 'Clump Thickness': '1 - 10', 'Uniformity of Cell Size': '1 - 10', 'Uniformity of Cell Shape': '1 - 10', 'Marginal Adhesion': '1 - 10', 'Single Epithelial Cell Size': '1 - 10', 'Bare Nuclei': '1 - 10', 'Bland Chromatin': '1 - 10', 'Normal Nucleoli': '1 - 10', 'Mitoses': '1 - 10', 'Class': '(2 for benign, 4 for malignant)'}