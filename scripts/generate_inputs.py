import numpy as np
from mlp.layers import LeakyReluLayer, ELULayer, SELULayer
import argparse

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')

parser.add_argument('--student_id', nargs="?", type=str, help='Your student id in the format "sxxxxxxx"')

args = parser.parse_args()

student_id = args.student_id

def generate_inputs(student_id):
    student_number = student_id
    tests = np.array([1.5, float(student_number[1:3]) / 10 - 5,
                      float(student_number[3:5]) / 10 - 5,
                      float(student_number[5:7]) / 10 - 5,
                      float(student_number[7]) / 10 - 5, -1.5])
    return tests



test_inputs = generate_inputs(student_id)
test_grads_wrt_outputs = np.array([[5., 10., -10., -5., 0., 10.]])

#produce leaky relu fprop and bprop
activation_layer = LeakyReluLayer()
leaky_relu_outputs = activation_layer.fprop(test_inputs)
leaky_relu_grads_wrt_inputs = activation_layer.bprop(
    test_inputs, leaky_relu_outputs, test_grads_wrt_outputs)

#produce ELU fprop and bprop
activation_layer = ELULayer()
ELU_outputs = activation_layer.fprop(test_inputs)
ELU_grads_wrt_inputs = activation_layer.bprop(
    test_inputs, ELU_outputs, test_grads_wrt_outputs)

#produce leaky relu fprop and bprop
activation_layer = SELULayer()
SELU_outputs = activation_layer.fprop(test_inputs)
SELU_grads_wrt_inputs = activation_layer.bprop(
    test_inputs, SELU_outputs, test_grads_wrt_outputs)

test_output = "Leaky ReLU:\nFprop: {}\nBprop: {}\nELU:\nFprop: {}\nBprop: {}\nSELU:\nFprop: {}\nBprop: {}\n"\
    .format(leaky_relu_outputs,
            leaky_relu_grads_wrt_inputs,
            ELU_outputs,
            ELU_grads_wrt_inputs, SELU_outputs, SELU_grads_wrt_inputs)

with open("{}_test_file.txt".format(student_id), "w+") as out_file:
    out_file.write(test_output)