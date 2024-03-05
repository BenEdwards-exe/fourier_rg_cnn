import tensorflow as tf

# kernels : H,W,C_in,C_out
# inputs  : N,H,W,C_in
@tf.function
def element_wise_multiply(input_data_real, input_data_imag, kernels_real, kernels_imag):
    input_shape = input_data_real.get_shape().as_list()

    output_list_real = []
    output_list_imag = []

    for index in range(input_shape[0]):
        # H,W,C_in -> H,W,C_in,1
        input_slice_real = tf.expand_dims(input_data_real[index], -1)
        input_slice_imag = tf.expand_dims(input_data_imag[index], -1)
        # Cross correlation (take conjugate)
        # (A + jB)(C - jD) = (AC + BD) + j(BC - AD)
        # A: inputs_real, jB: inputs_imag
        # C: filters_real, jD: filters_imag
        ac = tf.math.multiply(input_slice_real, kernels_real)
        bd = tf.math.multiply(input_slice_imag, kernels_imag)
        ad = tf.math.multiply(input_slice_real, kernels_imag)
        bc = tf.math.multiply(input_slice_imag, kernels_real)

        out_slice_real = ac + bd
        out_slice_imag = bc - ad

        out_slice_real = tf.reduce_sum(out_slice_real, 2)
        out_slice_imag = tf.reduce_sum(out_slice_imag, 2)

        output_list_real.append(out_slice_real)
        output_list_imag.append(out_slice_imag)

    out_real = tf.stack(output_list_real)
    out_imag = tf.stack(output_list_imag)

    return out_real, out_imag



class FourierConvLayer(tf.keras.layers.Layer):
    def __init__(self, input_feature_shape, output_channels, filter_size=3, use_bias=False, activation=None, **kwargs):
        super().__init__(**kwargs)

        self.input_feature_shape = input_feature_shape # BHWC (freq domain)
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.use_bias = use_bias

        # Filters in the freq domain
        # Create placeholder filters intialized to zero in case model is called and training=False
        freq_filter_shape = tf.TensorShape((self.input_feature_shape[1], self.input_feature_shape[2], self.input_feature_shape[3], self.output_channels))
        # self.filters_real = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=freq_filter_shape))
        # self.filters_imag = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=freq_filter_shape))
        self.filters_real = tf.zeros(shape=freq_filter_shape, dtype=tf.float32)
        self.filters_imag = tf.zeros(shape=freq_filter_shape, dtype=tf.float32)

        # Multiplication Layers
        self.mul_real = tf.keras.layers.Multiply()
        self.mul_imag = tf.keras.layers.Multiply()

        # Normalization Layers
        self.bnorm_real = tf.keras.layers.BatchNormalization()
        self.bnorm_imag = tf.keras.layers.BatchNormalization()

        # Activation Layers
        self.activation_real = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.activation_imag = tf.keras.layers.LeakyReLU(alpha=0.2)


        # Initialize Filters
        if self.filter_size >= self.input_feature_shape[1]:
            filter_shape = tf.TensorShape(
                (self.input_feature_shape[1], self.input_feature_shape[2], self.input_feature_shape[3], self.output_channels)
            )
            self.filters = self.add_weight(
                name="weight_spatial",
                shape=filter_shape,
                initializer=tf.keras.initializers.HeNormal(),
                regularizer='l2',
                trainable=True
            )
        else:
            filter_shape = tf.TensorShape(
                (self.filter_size, self.filter_size, self.input_feature_shape[3], self.output_channels)
            )
            self.filters = self.add_weight(
                name="weight_spatial",
                shape=filter_shape,
                initializer=tf.keras.initializers.HeNormal(),
                regularizer='l2',
                trainable=True
            )


    def call(self, inputs, training, isFeatFix=False):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        #BHWC
    
        # Only fft if training is being done
        if (training):
            # Pad the filters to the shape (H) of the inputs
            # if filters smaller than inputs.
            if self.filter_size < self.input_feature_shape[1]:
                padding_left = 0
                padding_right = self.input_feature_shape[1] - self.filter_size
                paddings = tf.constant([[padding_left, padding_right], [padding_left, padding_right], [0, 0], [0, 0]])
                filters_padded = tf.pad(self.filters, paddings, "CONSTANT", constant_values=0)
            else:
                filters_padded = self.filters

            # Transpose filters to have H,W as last 2 indicies
            filters_padded_trans = tf.transpose(filters_padded, [2, 3, 0, 1]) # H,W,C_in,C_out -> C_in,C_out,H,W
            # Cast filters to complex and fft
            filters_padded_trans = tf.cast(filters_padded_trans, tf.complex64)
            filters_freq = tf.signal.fft2d(filters_padded_trans)
            # Transpose filters back to original shape
            filters_freq = tf.transpose(filters_freq, [2, 3, 0, 1]) # C_in,C_out,H,W -> H,W,C_in,C_out

            # Seperate real and imaginary filter components
            self.filters_real = tf.math.real(filters_freq)
            self.filters_imag = tf.math.real(filters_freq)



        out_real, out_imag = element_wise_multiply(inputs_real, inputs_imag, self.filters_real, self.filters_imag)

        # Batch Norm
        out_real = self.bnorm_real(out_real, training=training)
        out_imag = self.bnorm_imag(out_imag, training=training)

        # Acivation function
        out_real = self.activation_real(out_real, training=training)
        out_imag = self.activation_imag(out_imag, training=training)

        return out_real, out_imag




class ComplexPoolLayer(tf.keras.layers.Layer):

    def __init__(self, pooling_window_size, feature_size, **kwargs):
        super().__init__(**kwargs)
        self.pooling_window_size = pooling_window_size
        self.feature_size = feature_size
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(pooling_window_size, pooling_window_size))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(pooling_window_size, pooling_window_size))

    def call(self, inputs, movingback, training):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        out_real = self.pool(inputs_real, training=training)
        out_imag = self.pool2(inputs_imag, training=training)

        return out_real, out_imag


class ComplexDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        # TODO: why is there no activation?
        self.fc_real = tf.keras.layers.Dense(self.n_neurons, activation=None)
        self.fc_imag = tf.keras.layers.Dense(self.n_neurons, activation=None)

        self.bnorm_real = tf.keras.layers.BatchNormalization()
        self.bnorm_imag = tf.keras.layers.BatchNormalization()

        self.relu_real = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu_imag = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        out_real = self.fc_real(inputs_real)
        out_imag = self.fc_imag(inputs_imag)

        out_real = self.bnorm_real(out_real, training=training)
        out_imag = self.bnorm_imag(out_imag, training=training)

        out_real = self.relu_real(out_real)
        out_imag = self.relu_imag(out_imag)

        return out_real, out_imag


class ComplexDropoutLayer(tf.keras.layers.Layer):
    def __init__(self, droprate, **kwargs):
        super().__init__(**kwargs)

        self.droprate = droprate
        self.mul_real = tf.keras.layers.Multiply()
        self.mul_imag = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        [batch_size, height, width, channels] = inputs_real.shape

        relu_ratio_real = tf.random.normal(shape=[batch_size, height, width, channels], mean=1.0, stddev= self.droprate/6)
        relu_ratio_imag = tf.random.normal(shape=[batch_size, height, width, channels], mean=1.0, stddev= self.droprate/6)

        if training:
            drop_ratio_real = tf.random.normal(shape=[batch_size, height, width, channels], mean=1.0, stddev=self.droprate/2)
            drop_ratio_imag = tf.random.normal(shape=[batch_size, height, width, channels], mean=1.0, stddev=self.droprate/2)

            out_real = self.mul_real([inputs_real, relu_ratio_real])
            out_real = self.mul_real([out_real, drop_ratio_real])

            out_imag = self.mul_imag([inputs_imag, relu_ratio_imag])
            out_imag = self.mul_imag([out_imag, drop_ratio_imag])

        else:
            # out_real = self.mul_real([inputs_real, relu_ratio_real])
            # out_imag = self.mul_imag([inputs_imag, relu_ratio_imag])
            out_real = inputs_real
            out_imag = inputs_imag

        return out_real, out_imag
    


class ComplexDenseDropoutLayer(tf.keras.layers.Layer):
    def __init__(self, droprate, **kwargs):
        super().__init__(**kwargs)

        self.droprate = droprate

        self.mul_real = tf.keras.layers.Multiply()
        self.mul_imag = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        if training:
            [batch_size, length] = inputs_real.shape

            drop_ratio_real = tf.random.normal(shape=(batch_size, length), mean=1.0, stddev=self.droprate/2)
            drop_ratio_imag = tf.random.normal(shape=(batch_size, length), mean=1.0, stddev=self.droprate/2)

            out_real = self.mul_real([inputs_real, drop_ratio_real])
            out_imag = self.mul_imag([inputs_imag, drop_ratio_imag])

        else:
            out_real = inputs_real
            out_imag = inputs_imag

        return out_real, out_imag
    


class ComplexConcatLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs):
        inputs_real = inputs[0]
        inputs_imag = inputs[1]

        out = self.concat([inputs_real, inputs_imag])

        return out
    
class FourierModel(tf.keras.models.Model):

    def __init__(self, batch_size, **kwargs):
        super().__init__(kwargs)

        self.num_classes = 3
        self.batch_size = batch_size
        # self.output_channel = [64, 128, 256, 512, 512]
        self.output_channel = [32, 32, 64, 128, 256]
        # self.output_channel = [32, 32, 64, 128, 256]
        self.fourier_layer_droprate = 0.5
        self.fully_connected_droprate = 0.5


        ## --- Input Layers --- ##
        self.input_real = tf.keras.layers.InputLayer(input_shape=(150, 150, 1))
        self.input_imag = tf.keras.layers.InputLayer(input_shape=(150, 150, 1))


        ## --- Block 1 --- ##
        self.fourier_conv_1 = FourierConvLayer([self.batch_size, 150, 150, 1], self.output_channel[0], filter_size=11)
        self.dropout_1 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        self.pooling_1 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 150, 150, self.output_channel[0]])
        # self.pooling_1 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 75, 75, self.output_channel[0]])


        ## --- Block 2 --- ##
        self.fourier_conv_2 = FourierConvLayer([self.batch_size, 75, 75, self.output_channel[0]], self.output_channel[1], filter_size=7)
        # self.fourier_conv_2 = FourierConvLayer([self.batch_size, 74, 74, self.output_channel[0]], self.output_channel[1])
        self.dropout_2 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        self.pooling_2 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 75, 75, self.output_channel[1]])
        # self.pooling_2 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 37, 37, self.output_channel[1]])


        ## --- Block 3 --- ##
        self.fourier_conv_3 = FourierConvLayer([self.batch_size, 37, 37, self.output_channel[1]], self.output_channel[2], filter_size=5)
        # self.fourier_conv_3 = FourierConvLayer([self.batch_size, 36, 36, self.output_channel[1]], self.output_channel[2])
        self.dropout_3 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        self.pooling_3 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 37, 37, self.output_channel[2]])
        # self.pooling_3 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 18, 18, self.output_channel[2]])


        ## --- Block 4 --- ##
        self.fourier_conv_4 = FourierConvLayer([self.batch_size, 18, 18, self.output_channel[2]], self.output_channel[3])
        self.dropout_4 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        self.pooling_4 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 18, 18, self.output_channel[3]])
        # self.pooling_4 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 9, 9, self.output_channel[3]])



        ## --- Block 5 --- ##
        self.fourier_conv_5 = FourierConvLayer([self.batch_size, 9, 9, self.output_channel[3]], self.output_channel[4])
        # self.fourier_conv_5 = FourierConvLayer([self.batch_size, 8, 8, self.output_channel[3]], self.output_channel[4])
        self.dropout_5 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        self.pooling_5 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 9, 9, self.output_channel[4]])
        # self.pooling_5 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 4, 4, self.output_channel[4]])


        # ## --- Block 6 --- ##
        # self.fourier_conv_6 = FourierConvLayer([self.batch_size, 4, 4, self.output_channel[4]], self.output_channel[4])
        # self.dropout_6 = ComplexDropoutLayer(droprate=self.fourier_layer_droprate)
        # self.pooling_6 = ComplexPoolLayer(pooling_window_size=2, feature_size=[self.batch_size, 4, 4, self.output_channel[4]])



        ## --- Flatten --- ##
        self.flatten_1 = tf.keras.layers.Flatten()
        self.flatten_2 = tf.keras.layers.Flatten()


        ## --- Fully Connected --- ##
        # -- FC1
        self.fully_connected_1 = ComplexDenseLayer(n_neurons=512)
        self.fully_connected_1_dropout = ComplexDenseDropoutLayer(droprate=self.fully_connected_droprate)
        # -- FC2
        self.fully_connected_2 = ComplexDenseLayer(n_neurons=256)
        self.fully_connected_2_dropout = ComplexDenseDropoutLayer(droprate=self.fully_connected_droprate)
        # -- FC3 
        self.fully_connected_3 = ComplexDenseLayer(n_neurons=256)
        self.fully_connected_3_dropout = ComplexDenseDropoutLayer(droprate=self.fully_connected_droprate)

        # -- Output Layer
        self.concat = ComplexConcatLayer()
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation="softmax")


    def call(self, inputs, training=None):
        inputs_real, inputs_imag = inputs


        ## --- Input Layers --- ##
        x_real = self.input_real(inputs_real)
        x_imag = self.input_imag(inputs_imag)

        ## --- Block 1 --- ##
        x_real, x_imag = self.fourier_conv_1([x_real, x_imag], training=training)
        x_real, x_imag = self.dropout_1([x_real, x_imag], training=training)
        x_real, x_imag = self.pooling_1([x_real, x_imag], movingback=True, training=training)

        ## --- Block 2 --- ##
        x_real, x_imag = self.fourier_conv_2([x_real, x_imag], training=training)
        x_real, x_imag = self.dropout_2([x_real, x_imag], training=training)
        x_real, x_imag = self.pooling_2([x_real, x_imag], movingback=True, training=training)

        ## --- Block 3 --- ##
        x_real, x_imag = self.fourier_conv_3([x_real, x_imag], training=training)
        x_real, x_imag = self.dropout_3([x_real, x_imag], training=training)
        x_real, x_imag = self.pooling_3([x_real, x_imag], movingback=True, training=training)

        ## --- Block 4 --- ##
        x_real, x_imag = self.fourier_conv_4([x_real, x_imag], training=training)
        x_real, x_imag = self.dropout_4([x_real, x_imag], training=training)
        x_real, x_imag = self.pooling_4([x_real, x_imag], movingback=True, training=training)

        ## --- Block 5 --- ##
        x_real, x_imag = self.fourier_conv_5([x_real, x_imag], training=training)
        x_real, x_imag = self.dropout_5([x_real, x_imag], training=training)
        x_real, x_imag = self.pooling_5([x_real, x_imag], movingback=True, training=training)

        ## --- Flatten --- ##
        x_real = self.flatten_1(x_real)
        x_imag = self.flatten_2(x_imag)

        ## --- Fully Connected --- ##
        # -- FC1
        x_real, x_imag = self.fully_connected_1([x_real, x_imag], training=training)
        x_real, x_imag = self.fully_connected_1_dropout([x_real, x_imag], training=training)
        # -- FC2
        x_real, x_imag = self.fully_connected_2([x_real, x_imag], training=training)
        x_real, x_imag = self.fully_connected_2_dropout([x_real, x_imag], training=training)
        # -- FC3
        x_real, x_imag = self.fully_connected_3([x_real, x_imag], training=training)
        x_real, x_imag = self.fully_connected_3_dropout([x_real, x_imag], training=training)

        # -- Output Layer
        x_concat = self.concat([x_real, x_imag])
        result = self.output_layer(x_concat)

        return result
    

if __name__ == "__main__":
    pass