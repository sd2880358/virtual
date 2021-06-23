from libsvm.svmutil import *
import pandas as pd

y, x = svm_read_problem('./australian_scale')
data = pd.DataFrame([x,y], dtype="float32")
data.to_csv("./convert_australian_scale.csv")
