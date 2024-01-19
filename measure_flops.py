import tensorflow as tf
from fourier_model import FourierModel
from rg_cnn_models import Toothless, FR_DEEP, RG_ZOO




def print_fourier_model_flops():

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    run_meta = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
    

        ## -- Start of running model
        batch_size = 32 # unfortunately batch size still needs to be declared
        model = FourierModel(batch_size=batch_size)

        input_shape = (1, 150, 150, 1)
        input_real = tf.compat.v1.Variable(dtype=tf.float32, initial_value=tf.zeros(input_shape))
        input_imag = tf.compat.v1.Variable(dtype=tf.float32, initial_value=tf.zeros(input_shape))
        inputs = (input_real, input_imag)

        output = model.call(inputs=inputs, training=False)
        ## -- End of running model


        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("Fourier Model Flops: {:,} --- params: {:,}".format(flops.total_float_ops, params.total_parameters))



def print_flops_toothless():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()



    run_meta = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # K.set_session(sess) [Use line below]
        tf.compat.v1.keras.backend.set_session(sess)
    
        model = Toothless(input_shape=(150, 150, 1), output_classes=3)
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # model.summary()

        test_input = tf.compat.v1.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=(1, *(150, 150, 1))))
        model.call(inputs=test_input, training=False)


        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("flops: {:,} --- params: {:,}".format(flops.total_float_ops, params.total_parameters))



def print_flops_fr_deep(input_shape=(150, 150, 1), output_classes=3):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()


    run_meta = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # K.set_session(sess) [Use line below]
        tf.compat.v1.keras.backend.set_session(sess)
    
        model = FR_DEEP(input_shape=input_shape, num_classes=output_classes)
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # model.summary()

        test_input = tf.compat.v1.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=(1, *input_shape)))
        model.call(inputs=test_input, training=False)


        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("flops: {:,} --- params: {:,}".format(flops.total_float_ops, params.total_parameters))


def print_flops_rg_zoo(input_shape=(150, 150, 1), output_classes=3):
    # tf.compat.v1.enable_eager_execution()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()


    run_meta = tf.compat.v1.RunMetadata()
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # K.set_session(sess) [Use line below]
        tf.compat.v1.keras.backend.set_session(sess)
    
        model = RG_ZOO(input_shape=input_shape, num_classes=output_classes)
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
        # model.summary()

        test_input = tf.compat.v1.Variable(dtype=tf.float32, initial_value=tf.zeros(shape=(1, *input_shape)))
        model.call(inputs=test_input, training=False)


        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("flops: {:,} --- params: {:,}".format(flops.total_float_ops, params.total_parameters))




if __name__ == "__main__":
    print_fourier_model_flops()

    print_flops_toothless()

    print_flops_fr_deep()

    print_flops_rg_zoo()