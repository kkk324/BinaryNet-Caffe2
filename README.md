# BNN_Caffe2
Quantized Neural Network on Caffe2

Goal:
Porting QNN(BNN) from Theano version to Caffe2 version.


What I Do
1.Test some Caffe2 ops using brew, 
  
  for example, in MNIST_BNN.ipynb:
  
  # NxCxHxW: 1x1x28x28 -> 1x1x28x28
  drop1 = brew.dropout(model,data, 'drop1', ratio=0.5, is_test=0)
  # NxCxHxW: 1x1x28x28 -> 20x1x28x28
  fc2 = brew.fc(model, drop1, 'fc2', dim_in=1 * 28 * 28, dim_out=20)
  bn2 = brew.spatial_bn(model, fc2, 'bn2',dim_in = 20) 

  see https://zhuanlan.zhihu.com/p/27096092

2.Key parts: build binarization functions

  1. Understand and test binarization functions of Theano version (I'm here now)
  2. Porting




Refs.
QNN Theano version paper
https://arxiv.org/abs/1609.07061 

Theano version github
https://github.com/MatthieuCourbariaux/BinaryNet



