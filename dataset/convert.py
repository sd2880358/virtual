from libsvm.svmutil import *
import pandas as pd

y, x = svm_read_problem('./australian_scale')
print(x[0][2])
