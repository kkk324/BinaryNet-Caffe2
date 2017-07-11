from caffe2.python import core, workspace
import numpy as np
import binary_tool as bt

'''
for implement binary_net on caffe2
original theano version would be in the upper comment
'''

#Generate an array with random number elements 
X = np.random.randn(2, 3).astype(np.float32)
print("Generated X from numpy:\n{}".format(X))
workspace.FeedBlob("X", X)

X = bt.tt(X)
print("Generated X from tt(X):\n{}".format(X))

X = bt.hard_sigmoid(X)
print("Generated X from hard_sigmoid(X):\n{}".format(X))
