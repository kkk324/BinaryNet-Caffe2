import numpy as np
import os
import shutil
import utils
import models

from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")

#################
# Prepare data

utils.PrepareDataset()


############################
# Start constructing models

arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)

data, label = models.AddInput(
    train_model, batch_size=64,
    db=os.path.join(utils.data_folder, 'mnist-train-nchw-leveldb'),
    db_type='leveldb')
#softmax = models.AddLeNetModel(train_model, data)
softmax = models.AddMLP(train_model, data, batch_size=64)
models.AddTrainingOperators(train_model, softmax, label)
models.AddBookkeepingOperators(train_model)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = model_helper.ModelHelper(
    name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label = models.AddInput(
    test_model, batch_size=100,
    db=os.path.join(utils.data_folder, 'mnist-test-nchw-leveldb'),
    db_type='leveldb')
#softmax = models.AddLeNetModel(test_model, data)
softmax = models.AddMLP(test_model, data, batch_size=100)
models.AddAccuracy(test_model, softmax, label)

#print(str(train_model.param_init_net.Proto()) + '\n...')
#print(str(test_model.param_init_net.Proto()) + '\n...')


print('Training...')
# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net, overwrite=True)




# set the number of iterations and track the accuracy & loss
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations. 

data_array = []
drop1_array = []
fc2_array = []

for i in range(total_iters):
#for i in range(1):
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    print('iter {0} loss = {1} '.format(i, loss[i]))
    print('         accuracy = {0} '.format(accuracy[i]))
    
    print("Current blobs in the workspace: {}".format(workspace.Blobs()))

    print("Workspace has blob 'data'? {}".format(workspace.HasBlob("data")))
    #print("Fetched data:\n{}".format(workspace.FetchBlob("data")))
    data_array.append(workspace.FetchBlob("data"))
    print('data_array',np.shape(data_array))


    print("Workspace has blob 'drop1'? {}".format(workspace.HasBlob("drop1")))
    #print("Fetched drop1:\n{}".format(workspace.FetchBlob("drop1")))
    drop1_array.append(workspace.FetchBlob("drop1"))
    print('drop1_array',np.shape(drop1_array))

    print("Workspace has blob 'fc2'? {}".format(workspace.HasBlob("fc2")))
    #print("Fetched fc2:\n{}".format(workspace.FetchBlob("fc2")))
    fc2_array.append(workspace.FetchBlob("fc2"))
    print('fc2_array',np.shape(fc2_array))
 
# store params of train_model
train_params = {p: workspace.FetchBlob(p) for p in train_model.GetParams()}

# run a test pass on the test net
print('Testing...')
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
test_accuracy = np.zeros(100)

# set params of test_model according to the stored train_params
for p in train_params:
    print(p)
    workspace.FeedBlob(p, train_params[p])

test_iter = 100
for i in range(test_iter):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')

print('average test accuracy = {0}'.format(np.mean(test_accuracy)))


