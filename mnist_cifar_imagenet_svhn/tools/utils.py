from collections import defaultdict
import random
import numpy as np
from keras.models import Model


class BaseCoverage:

    def __init__(self, model):
        self.model = model
        self.layer_names = [layer.name
                            for layer in self.model.layers
                            if 'flatten' not in layer.name and 'input' not in layer.name]
        self.intermediate_layer_model = Model(inputs=self.model.input,
                                              outputs=[self.model.get_layer(layer_name).output
                                                       for layer_name in self.layer_names])
        self.model_layer_dict = self.init_coverage_tables()

    def init_coverage_tables(self):
        model_layer_dict = defaultdict(bool)
        for layer in self.model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False
        return model_layer_dict

    def neuron_to_cover(self):
        not_covered = [(layer_name, index) for (layer_name, index), v in self.model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self.model_layer_dict.keys())
        return layer_name, index

    def neuron_covered(self):
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        total_neurons = len(self.model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, **kwargs):
        pass

    def scale(self, intermediate_layer_output, rmax=1, rmin=0):
        """standardized"""
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def final_coverage(self, **kwargs):
        cov_name = self.__class__.__name__
        print("Coverage Type: {}".format(cov_name))
        cur_percent = 0.0
        inputs = kwargs['inputs']
        threshold = kwargs['threshold']
        K_value = kwargs['K']
        for idx, x in enumerate(inputs):
            x = np.expand_dims(x, axis=0)
            self.update_coverage(input_data=x, threshold=threshold,K=K_value)
            activate_neurons, total_neurons, rate = self.neuron_covered()
            if rate != cur_percent:
                cur_percent = rate
                # print(idx, activate_neurons, total_neurons, cur_percent)
        print(cov_name,cur_percent)
        return cur_percent


class NeuronCoverage(BaseCoverage):
    def __init__(self, model):
        BaseCoverage.__init__(self, model=model)

    def update_coverage(self, **kwargs):
        input_data = kwargs['input_data']
        threshold = kwargs['threshold']

        intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self.scale(intermediate_layer_output[0])
            for num_neuron in range(scaled.shape[-1]):
                scaled_output = scaled[..., num_neuron]
                mean = np.mean(scaled_output)
                if mean > threshold and not self.model_layer_dict[(self.layer_names[i], num_neuron)]:
                    self.model_layer_dict[(self.layer_names[i], num_neuron)] = True


class TopKNeuronCoverage(BaseCoverage):
    def __init__(self, model):
        BaseCoverage.__init__(self, model=model)

    def update_coverage(self, **kwargs):

        input_data = kwargs['input_data']
        K_value = kwargs['K']
        intermediate_layer_outputs = self.intermediate_layer_model.predict(input_data)

        # compute coverage for i-th layer
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            scaled = self.scale(intermediate_layer_output[0])
            neurons_list = list()
            # compute output for each channel
            for num_neuron in range(scaled.shape[-1]):
                scaled_output = scaled[..., num_neuron]
                mean = np.mean(scaled_output)
                neurons_list.append(mean)
            # get top-k neurons rank
            neurons_rank = np.argsort(neurons_list)
            for j in range(1,K_value+1):
                # get top j-th neuron index
                neurons_index = neurons_rank[-j]
                if not self.model_layer_dict[(self.layer_names[i], neurons_index)]:
                    self.model_layer_dict[(self.layer_names[i], neurons_index)] = True
