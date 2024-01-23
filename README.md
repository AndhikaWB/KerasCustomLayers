# Keras Custom Layers
Simple custom layers that are not implemented yet on Keras (or at least not directly).

- `ActiveGaussianNoise`

    Based on `GaussianNoise`, but will always be active even when not training (e.g. when testing/predicting). Useful if you want to visualize the noise output since it was not possible without this modification.

- `MinPoolingXD`

    Based on `MaxPoolingXD`, but with negative input and output. Available for 1D, 2D, and 3D data (just use the correct class name). The global versions are also available (`GlobalMinPoolingXD`).

- `ImprovedUpSampling1D`

    Based on `UpSampling2D`, but reworked for one dimension (time series) data. Multiple interpolation modes are supported just like the 2D version (e.g. bicubic, bilinear).

- `ScaledDiscretization`

    Based on `Discretization`, but with output index ranged from 0-1 just like when using min-max scaler. Only int mode is supported, but the output will be float (scaled). **Note:** make sure the input is already scaled using min-max scaler if you want to compare it with the result. And just like the original layer, you may still need to call `adapt` first.

    Example code:
    ``` python
    discrete_layer = ScaledDiscretization(num_bins = 20)
    discrete_layer.adapt(train_data)
    ```

- `TrainingOnly`

    Enable a layer only when in training phase. Useful if you want to apply smoothing (e.g. pooling and upsampling) on train data only. Input should be another layer.

- `RandomApply`

    Randomly apply a layer, with configurable rate and seed. Rate should be float with range from 0-1
    
    If rate is positive (e.g. 0.2), then the layer will be used if RNG output is between 0.0 - 0.2. If the rate is negative (e.g. -0.2), then the layer will be applied if the RNG output is between 0.81 - 1.0. This may be useful if you want to use 2 opposing layers that shouldn't be active at the same time.

    Example code:
    ``` python
    # Both of these layers will be applied together when active (same seed)
    RandomApply(
        TrainingOnly(AveragePooling1D(pool_size = 3, padding = 'same')),
        rate = 0.4, seed = 1234
    ),
    RandomApply(
        TrainingOnly(ImprovedUpSampling1D(size = 3, interpolation = 'bicubic')),
        rate = 0.4, seed = 1234
    ),

    # This layer will only be applied if the previous layers are not active
    RandomApply(
        TrainingOnly(discrete_layer),
        rate = -0.2, seed = 1234
    ),
    ```

- `IterativeSwitch`

    Switch between all wrapped layers iteratively (input should be a list of layers).

    Example code:
    ``` python
    # Switch between min pooling and max pooling
    IterativeSwitch([
        MinPooling1D(pool_size = 3, padding = 'same'),
        MaxPooling1D(pool_size = 3, padding = 'same')
    ]),

    ImprovedUpSampling1D(size = 3, interpolation = 'bicubic')
    ```

- `DoNothing`

    A layer that does nothing (input = output). Can be used together with `IterativeSwitch` to allow some of the original data to passthrough.