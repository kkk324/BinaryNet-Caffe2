import os
import numpy as np
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

def AddLeNetModel(model, data):
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddMLP(model, data, batch_size):
    '''
    Implement MLP model on MNIST
    '''
    num_units = 4096 # number of nuerons in fc layer
    num_labels = 10 # for 10 classes in mnist
    drop1 = brew.dropout(model, data, 'drop1', ratio=0.5, is_test=0)

    fc2 = brew.fc(model, drop1, 'fc2', dim_in=1 * 28 * 28, dim_out=num_units)
    model.Reshape([fc2], [fc2, 'fc2_old_shape'], shape=(batch_size, num_units, 1, 1))
    bn2 = brew.spatial_bn(model, fc2, 'bn2', num_units, epsilon=1e-4, momentum=0.9)
    relu2 = brew.relu(model, bn2, 'relu2')

    fc3 = brew.fc(model, relu2, 'fc3', dim_in=num_units, dim_out=num_units)
    model.Reshape([fc3], [fc3, 'fc3_old_shape'], shape=(batch_size, num_units, 1, 1))
    bn3 = brew.spatial_bn(model, fc3, 'bn3', num_units, epsilon=1e-4, momentum=0.9)
    relu3 = brew.relu(model, bn3, 'relu3')

    fc4 = brew.fc(model, relu3, 'fc4', dim_in=num_units, dim_out=num_units)
    model.Reshape([fc4], [fc4, 'fc4_old_shape'], shape=(batch_size, num_units, 1, 1))
    bn4 = brew.spatial_bn(model, fc4, 'bn4', num_units, epsilon=1e-4, momentum=0.9)
    relu4 = brew.relu(model, bn4, 'relu4')

    fc5 = brew.fc(model, relu4, 'fc5', dim_in=num_units, dim_out=num_labels)
    model.Reshape([fc5], [fc5, 'fc5_old_shape'], shape=(batch_size, num_labels, 1, 1))
    bn5 = brew.spatial_bn(model, fc5, 'bn5', num_labels, epsilon=1e-4, momentum=0.9)
    softmax = brew.softmax(model, bn5, 'softmax')
    return softmax


def AddMLP_BN(model, data):
    '''
    Implement MLP model on MNIST
    '''
    # number of nuerons in fc layer
    num_units = 4096
   
    # NCHW: 64 x 1 x 28 x 28 -> 64 x 1 x 28 x 28
    drop1 = brew.dropout(model, data, 'drop1', ratio=0.5, is_test=0)

    # NCHW: 64 x 1 x 28 x 28 -> 64 x 4096
    fc2 = brew.fc(model, drop1, 'fc2', dim_in=1 * 28 * 28, dim_out=num_units)
    
    
    fc2_reshaped = model.Reshape(
                                 [fc2],
                                 ['fc2_reshaped'],
                                 shape=(4,2)
    )
    

    # bn2 = brew.spatial_bn(model, fc2, 'bn2', num_units, epsilon=1e-3, momentum=0.9, is_test=is_test)
    #relu2 = brew.relu(model, fc2, 'relu2')

    #fc3 = brew.fc(model, relu2, 'fc3', dim_in=num_units, dim_out=num_units)
    # bn3 = brew.spatial_bn(model, fc3, 'bn3', bn_units, epsilon=1e-3, momentum=0.9, is_test=is_test)
    #relu3 = brew.relu(model, fc3, 'relu3')

    #fc4 = brew.fc(model, relu3, 'fc4', dim_in=num_units, dim_out=num_units)
    # bn4 = brew.spatial_bn(model, fc4, 'bn4', bn_units, epsilon=1e-3, momentum=0.9, is_test=is_test)
    #relu4 = brew.relu(model, fc4, 'relu4')

    #fc5 = brew.fc(model, relu4, 'fc5', dim_in=num_units, dim_out=10) # 10 for 10-classes
    # bn5 = brew.spatial_bn(model, fc5, 'bn5', 10, epsilon=1e-3, momentum=0.9, is_test=is_test)
    softmax = brew.softmax(model, fc2, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)

def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
model.Summarize(model.param_to_grad[param], [], to_file=1)
