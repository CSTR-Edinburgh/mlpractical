# from mlp.learning_rules import AdamLearningRuleWithWeightDecay
# from mlp.schedulers import CosineAnnealingWithWarmRestarts
from mlp.layers import LeakyReluLayer, ParametricReluLayer, RandomReluLayer, ExponentialLinearUnitLayer
import numpy as np
import os

def test_leaky_relu():
    # loaded = np.load("../data/correct_results.npz")
    rng = np.random.RandomState(92019)

    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'activation_debug_pack.npy'), allow_pickle=True).item()

    layer = LeakyReluLayer()

    out = layer.fprop(x)

    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))

    correct_outputs = correct_outputs['leaky_relu']

    fprop_test = np.allclose(correct_outputs['fprop_correct'], out)

    bprop_test = np.allclose(correct_outputs['grad_correct'], grads)

    return fprop_test, out, correct_outputs['fprop_correct'], bprop_test, grads, correct_outputs['grad_correct']


def test_random_relu():
    rng = np.random.RandomState(92019)

    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'activation_debug_pack.npy'), allow_pickle=True).item()

    layer = RandomReluLayer()

    out = layer.fprop(x, leakiness=correct_outputs['random_relu']['leakiness'])

    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))

    correct_outputs = correct_outputs['random_relu']

    fprop_test = np.allclose(correct_outputs['fprop_correct'], out)

    bprop_test = np.allclose(correct_outputs['grad_correct'], grads)

    return fprop_test, out, correct_outputs['fprop_correct'], bprop_test, grads, correct_outputs['grad_correct']


def test_parametric_relu():
    # loaded = np.load("../data/correct_results.npz")
    rng = np.random.RandomState(92019)
    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'activation_debug_pack.npy'), allow_pickle=True).item()

    layer = ParametricReluLayer(alpha=0.25)
    out = layer.fprop(x.copy())
    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))
    grad_wrt_param = layer.grads_wrt_params(inputs=x, grads_wrt_outputs=np.ones(x.shape))

    correct_outputs = correct_outputs['prelu']

    fprop_test = np.allclose(correct_outputs['fprop_correct'], out)

    bprop_test = np.allclose(correct_outputs['grad_correct'], grads)

    grad_wrt_param_test = np.allclose(correct_outputs['grad_param'], grad_wrt_param)

    return fprop_test, out, correct_outputs['fprop_correct'], bprop_test, grads, correct_outputs['grad_correct'], \
           grad_wrt_param_test, grad_wrt_param, correct_outputs['grad_param']


def test_exponential_linear_unit():
    rng = np.random.RandomState(92019)
    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'activation_debug_pack.npy'), allow_pickle=True).item()

    layer = ExponentialLinearUnitLayer()
    out = layer.fprop(x)
    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))

    correct_outputs = correct_outputs['elu']

    fprop_test = np.allclose(correct_outputs['fprop_correct'], out, rtol=1e-3, atol=1e-6)

    bprop_test = np.allclose(correct_outputs['grad_correct'], grads, rtol=1e-3, atol=1e-6)

    return fprop_test, out, correct_outputs['fprop_correct'], bprop_test, grads, correct_outputs['grad_correct']
