try: from tensorflow import keras
except ImportError: import keras
from keras.engine import base_layer
import tensorflow as tf
import copy

# By default, gaussian noise is only active on training mode
# This make it difficult to visualize the layer output
class ActiveGaussianNoise(keras.layers.GaussianNoise):
    '''GaussianNoise, but will always be active even when not training'''   
    def call(self, inputs):
        return super().call(inputs, training = True)
    
# --------------------

class MinPooling1D(keras.layers.MaxPooling1D):
    '''MaxPooling1D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

class MinPooling2D(keras.layers.MaxPooling2D):
    '''MaxPooling2D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

class MinPooling3D(keras.layers.MaxPooling3D):
    '''MaxPooling3D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

# --------------------

class GlobalMinPooling1D(keras.layers.GlobalMaxPooling1D):
    '''GlobalMaxPooling1D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

class GlobalMinPooling2D(keras.layers.GlobalMaxPooling2D):
    '''GlobalMaxPooling2D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

class GlobalMinPooling3D(keras.layers.GlobalMaxPooling3D):
    '''GlobalMaxPooling3D, but with negative input and output'''
    def call(self, inputs):
        return -super().call(-inputs)

# --------------------

class ImprovedUpSampling1D(keras.layers.UpSampling2D):
    '''UpSampling2D, but reworked for one dimension (time series) data'''
    def __init__(self, size = 2, interpolation = 'bicubic', **kwargs):
        if kwargs.get('data_format'): kwargs.pop('data_format')

        # Number of features as last dimension
        data_format = 'channels_last'
        # 1D data only has 1 row
        size = (1, size)

        super().__init__(
            size = size,
            data_format = data_format,
            interpolation = interpolation,
            **kwargs
        )

        self.input_spec = keras.layers.InputSpec(ndim = 3)
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(
            [input_shape[0], self.size[1] * input_shape[2], input_shape[3]]
        )
    
    def call(self, inputs):
        # Add row dimension, then remove it again
        inputs = tf.expand_dims(inputs, axis = 1)
        outputs = super().call(inputs)
        return tf.squeeze(outputs, axis = 1)

# It seems that discretization is using quantile
# It may not be too accurate unless boundaries is set manually
class ScaledDiscretization(keras.layers.Discretization):
    '''Discretization layer, but with output index ranged from 0-1 just like min-max scaler.
    Only int mode is supported, but the output will be float (scaled)'''
    
    def __init__(self, bin_boundaries = None, num_bins = None, epsilon = 0.01, sparse = False, **kwargs):
        # Output mode should always be int to avoid error
        if kwargs.get('output_mode'): kwargs.pop('output_mode')
        output_mode = 'int'

        super().__init__(
            bin_boundaries = bin_boundaries, num_bins = num_bins, epsilon = epsilon,
            output_mode = output_mode, sparse = sparse, **kwargs
        )

    def call(self, inputs):
        # There will be repeating zero at the beginning of array if "num_bins" is too big
        # We only need the last zero (assuming min value is zero) and all non-zero numbers after that
        bounds = [ i for i in self.bin_boundaries if i != 0.0 ]
        # Calculate the length, but include zero too
        num_bounds = len(bounds) + 1

        # Get how many buckets are there based on num of boundaries
        # The length of "bin_boundaries" is always one less than the "num_bins"
        # If "num_bins" is 20 then "bin_boundaries" is 19, so we need +1 somehow
        buckets = len(self.bin_boundaries) + 1 - num_bounds
        # -1 so the minimum index would be 0, not 1
        return (super().call(inputs) / buckets) - 1

# See gaussian noise or dropout layer for reference
# https://stackoverflow.com/questions/66983059/randomly-apply-keras-preprocessinglayer
class RandomApply(base_layer.BaseRandomLayer):
    '''Randomly apply a layer, with configurable rate and seed'''
    def __init__(self, layer: keras.layers.Layer, rate = 0.5, seed = None, **kwargs):
        super().__init__(seed = seed, **kwargs)
        self.layer = layer
        self.rate = rate
        self.seed = seed
    
    def build(self, input_shape):
        self.layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # When initializing a model, this function will be called with test (empty) input
        # In that case, we should call the original layer to adapt the test input (100% chance)
        # If not adapted, some parameters like input/output shape can't be determined
        if not tf.executing_eagerly():
            return self.layer(inputs)

        # Generate random number between 0 and 1 (inclusive)
        random = self._random_generator.random_uniform([])
        # Use negative rate value to invert the operator
        # Useful to prevent 2 opposing layers active at the same time
        if self.rate < 0:
            self.rate = self.rate * -1
            condition = random > self.rate
        else:
            condition = random <= self.rate

        outputs = tf.cond(
            pred = condition,
            true_fn = lambda: self.layer(inputs),
            false_fn = lambda: inputs
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer': keras.layers.serialize(self.layer),
            'rate': self.rate,
            'seed': self.seed
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config['layer'] = keras.layers.deserialize(config['layer'])
        layer = cls(**config)
        return layer

    @property
    def input_shape(self):
        return self.layer.input_shape
    
    @property
    def input_spec(self):
        return self.layer.input_spec

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

# See bidirectional layer for reference
class IterativeSwitch(keras.layers.Layer):
    '''Switch between all wrapped layers iteratively (input should be a list of layers)'''
    def __init__(self, layers: list, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.len_layer = len(layers)
        self.cur_layer = 0
    
    def build(self, input_shape):
        # Access index since I'm not sure whether it's referencing value or address
        # If using index, it's most likely referencing address (original object, not a copy)
        for i in range(self.len_layer): self.layers[i].build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        if not tf.executing_eagerly():
            # Adapt on every layers, but return only the output of first layer
            for i in range(self.len_layer): self.layers[i](inputs)
            return self.layers[0](inputs)

        outputs = self.layers[self.cur_layer](inputs)
        self.cur_layer = (self.cur_layer + 1) % self.len_layer
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'layers': [ keras.layers.serialize(layer) for layer in self.layers ]
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config['layers'] = [ keras.layers.deserialize(layer) for layer in config['layers'] ]
        layer = cls(**config)
        return layer
    
    @property
    def input_shape(self):
        return self.layers[self.cur_layer].input_shape
    
    @property
    def input_spec(self):
        return self.layers[self.cur_layer].input_spec

    def compute_output_shape(self, input_shape):
        return self.layers[self.cur_layer].compute_output_shape(input_shape)

class TrainingOnly(keras.layers.Layer):
    '''Enable a layer only when in training phase'''
    def __init__(self, layer: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs, training = None):
        if not tf.executing_eagerly():
            return self.layer(inputs)
        
        return keras.backend.in_train_phase(
            self.layer(inputs),
            inputs,
            training
        )

    def get_config(self):
        config = super().get_config()
        config.update({'layer': keras.layers.serialize(self.layer)})
        return config
    
    @classmethod
    def from_config(cls, config):
        config = copy.deepcopy(config)
        config['layer'] = keras.layers.deserialize(config['layer'])
        layer = cls(**config)
        return layer

    @property
    def input_shape(self):
        return self.layer.input_shape
    
    @property
    def input_spec(self):
        return self.layer.input_spec

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

class DoNothing(keras.layers.Layer):
    '''A layer that does nothing (input = output)'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    def call(self, inputs):
        return inputs
 
    def compute_output_shape(self, input_shape):
        return input_shape