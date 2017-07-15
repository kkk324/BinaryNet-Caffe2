'''

GOAL : verify the functionality/correctness of operation/layer
Testing methodology : compare the output of operation/layer to Theano's

Preliminary, use the static generated data for comparison.
Note: Theano adopts symbolic computing, intermediate values in a computation cannot be printed
in the normal python way with the print statement, because Theano has no statements.
Instead there is the 'Print' Op.
http://deeplearning.net/software/theano/library/printing.html

input =
[[8.53484750e-01   1.00000000e+00   0.00000000e+00   9.02203321e-01
6.46270394e-01   9.61881950e-02   7.77260303e-01   3.87970775e-01 ...

expected =
[[1.  1.  0.  1.  1.  0.  1.  0.  0.  0.  1.  1.  0.  1.  0.  1.  1.  1. ...

'''
import sys
sys.path.append("../src")

import binary_tool
import numpy as np
import re

def flatten_from_file(fd):
    array = []
    for line in fd:
        for x in re.split(r'\[|\]|\s*', line):
            if not x == '':
                array.append(float(x))
    return array

# fn : testing function
# fracton : there is a little bit precision different between Theano and Caffe
def comparison(fn, input_fd_name, expected_fd_name, fraction):
    # Reading the data from file and flat to 1d list
    f_input = open(input_fd_name, "r")
    inputs = flatten_from_file(f_input)

    inputs = np.array(inputs).astype(np.float32)
    outputs = fn(inputs)

    f_expected = open(expected_fd_name, "r")
    expecteds = flatten_from_file(f_expected)

    # Perform the element-wise comparison one by one
    assert len(outputs) == len(expecteds), "len(input) = %r, len(expecteds) = %r" \
        % (len(outputs), len(expecteds))

    for i in range(0, len(outputs)):
        out = round(outputs[i], fraction)
        exp = round(expecteds[i], fraction)
        assert out == exp, "input = %r, expected = %r" % (out, exp)

# def test_round3():
#     comparison(binary_tool.round3, 'input.txt', 'expected.txt', 5)

# test_round3()

def test_hard_sigmoid():
    # test failed against fraction above 5
    comparison(binary_tool.hard_sigmoid, 'clip_input.txt', 'clip_output.txt', 5)

test_hard_sigmoid()

# TODO if we has a round3 implementation
# static test : by file format
# def test_round3():
#     generated_from_caffe2 = round3(input_from_theano)  // invoke round3()
#     comparison(generated_from_caffe2, generated_from_theano)
#
# def test_xxx(): ...

'''
output = []
for x in range(0, 100):
    temp = []
    for y in range(0, 4096):
        temp.append(array.pop(0))
    output.append(temp)

# output_file = open("test.txt", "a")
output_file = open("test2.txt", "a")
output_file.write(str(output))
output_file.close()
'''
