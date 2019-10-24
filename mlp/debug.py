import ipdb
def grads_wrt_params(self, activations, grads_wrt_outputs):
        """Calculates gradients with respect to the model parameters.
        Args:
            activations: List of all activations from forward pass through
                model using `fprop`.
            grads_wrt_outputs: Gradient with respect to the model outputs of
               the scalar function parameter gradients are being calculated
               for.
        Returns:
            List of gradients of the scalar function with respect to all model
            parameters.
        """
        grads_wrt_params = []
        for i, layer in enumerate(self.layers[::-1]):
            inputs = activations[-i - 2]
            outputs = activations[-i - 1]
            grads_wrt_inputs = layer.bprop(inputs, outputs, grads_wrt_outputs)
            ipdb.set_trace() ################## Break Point ######################
            if isinstance(layer, LayerWithParameters) or isinstance(layer, StochasticLayerWithParameters):
                grads_wrt_params += layer.grads_wrt_params(inputs, grads_wrt_outputs)[::-1]
            grads_wrt_outputs = grads_wrt_inputs
        return grads_wrt_params[::-1]
