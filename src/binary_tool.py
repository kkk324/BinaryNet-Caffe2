from caffe2.python import core, workspace

'''
for implement binary_net on caffe2
original theano version would be in the upper comment
'''

'''
def hard_sigmoid(x):
	return clip((x+1.)/2.,0,1)
'''

def tt(x):
    workspace.FeedBlob("x", x)
    Clip = core.CreateOperator(
        "Clip",
        ["x"],
        ["x"],
        min=0.,
        max=1.
    )
    workspace.RunOperatorOnce(Clip)
    return workspace.FetchBlob("x")


def hard_sigmoid(x):
    x = (x + 1.) / 2.
    workspace.FeedBlob("x", x)
    Clip = core.CreateOperator(
        "Clip",
        ["x"],
        ["x"],
        min=0.,
        max=1.
    )
    workspace.RunOperatorOnce(Clip)
    return workspace.FetchBlob("x")


'''
def binary_tanh_unit(x):
    return 2.*round(hard_sigmoid(x))-1.
'''
#TODO: implement binary_tanh_unit

'''
def binary_sigmoid_unit(x):
    return round(hard_sigmoid(x))
'''
#TODO: implement binary_sigmoid_unit

