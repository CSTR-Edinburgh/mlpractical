import argparse
import os
import numpy as np

from mlp.layers import LeakyReluLayer, RandomReluLayer, ExponentialLinearUnitLayer, ParametricReluLayer

parser = argparse.ArgumentParser(description='Welcome to Conv test script')

parser.add_argument('--student_id', nargs="?", type=str, help='Your student id in the format "sxxxxxxx"')

args = parser.parse_args()

student_id = args.student_id


def fprop_bprop_layer(inputs, activation_layer, grads_wrt_outputs, weights, params=False):
    if params:
        activation_layer.params = [weights]

    fprop = activation_layer.fprop(inputs)
    bprop = activation_layer.bprop(inputs, fprop, grads_wrt_outputs)

    outputs = [fprop, bprop]
    if params:
        grads_wrt_weights = activation_layer.grads_wrt_params(inputs, grads_wrt_outputs)
        outputs.append(grads_wrt_weights)

    return outputs


def get_student_seed(student_id):
    student_seed_number = int(student_id[1:])
    return student_seed_number


seed = get_student_seed(student_id)
rng = np.random.RandomState(seed)

output_dict = dict()

inputs = rng.normal(loc=0.0, scale=1.0, size=(32, 3, 8, 8))
grads_wrt_outputs = rng.normal(loc=0.0, scale=1.0, size=(32, 3, 8, 8))
weights = rng.normal(loc=0.0, scale=1.0, size=(1))

output_dict['inputs'] = inputs
output_dict['weights'] = weights
output_dict['grads_wrt_outputs'] = grads_wrt_outputs

for activation_layer, params_flag in zip(
        [LeakyReluLayer, ParametricReluLayer, RandomReluLayer, ExponentialLinearUnitLayer],
        [False, True, False, False]):
    outputs = fprop_bprop_layer(inputs, activation_layer(), grads_wrt_outputs, weights, params_flag)
    output_dict['{}_{}'.format(activation_layer.__name__, 'fprop')] = outputs[0]
    output_dict['{}_{}'.format(activation_layer.__name__, 'bprop')] = outputs[1]
    if params_flag:
        output_dict['{}_{}'.format(activation_layer.__name__, 'grads_wrt_outputs')] = outputs[2]

np.save('/home/antreas/mlpractical_dev/data/test_data.npy', output_dict)

test_data = np.load(os.path.join(os.environ['MLP_DATA_DIR'], '{}_activation_test_pack.npy'.format(seed)), allow_pickle=True)

for key, value in test_data.item().items():
    print(key, value)