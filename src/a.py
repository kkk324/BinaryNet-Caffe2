import numpy as np
import os

a = np.arange(6)
b = a.reshape((3,-1))
#.reshape((3,-1))
current_folder = os.path.join(os.path.expanduser('~'), 'BNN_Caffe2/log')

print(a)
print(b)

print(current_folder)
