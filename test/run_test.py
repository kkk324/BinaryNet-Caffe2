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

import re

def flatten_from_file(fd):
    array = []
    for line in fd:
        for x in re.split(r'\[|\]|\s*', line):
            if not x == '':
                array.append(float(x))
    return array

def comparison(input_fd_name, expected_fd_name):
    # Reading the data from file and flat to 1d list
    f_input = open(input_fd_name, "r")
    inputs = flatten_from_file(f_input)

    f_expected = open(expected_fd_name, "r")
    expecteds = flatten_from_file(f_expected)

    # Perform the element-wise comparison one by one
    assert len(inputs) == len(expecteds), "len(input) = %r, len(expecteds) = %r" \
        % (len(inputs), len(expecteds))

    for i in range(0, len(inputs)):
        assert inputs[i] == expecteds[i], "input = %r, expected = %r" \
            % (inputs[i], expecteds[i])

def test_round3():
    comparison('input.txt', 'expected.txt')

test_round3()

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
