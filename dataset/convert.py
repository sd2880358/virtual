from libsvm.svmutil import *
import pandas as pd

y, x = svm_read_problem('./australian_scale')
features = [[i for i in x[j]] for j in range(len(x[0]))]
data = pd.DataFrame([features, y], dtype='float32')
data.to_csv("./c_australian_scale")
