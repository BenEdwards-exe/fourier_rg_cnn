import numpy as np
import tensorflow as tf


# Read in weights of a conv layer and generate text file with fft'd valued
def conv_layer_weights():

    filter_weights = np.load("C_Implementation/trained_weights/weight_spatial_0.npy")
    fft_weights_and_save(filter_weights, 1)
    filter_weights = np.load("C_Implementation/trained_weights/weight_spatial_1.npy")
    fft_weights_and_save(filter_weights, 2)
    filter_weights = np.load("C_Implementation/trained_weights/weight_spatial_2.npy")
    fft_weights_and_save(filter_weights, 3)
    filter_weights = np.load("C_Implementation/trained_weights/weight_spatial_3.npy")
    fft_weights_and_save(filter_weights, 4)
    filter_weights = np.load("C_Implementation/trained_weights/weight_spatial_4.npy")
    fft_weights_and_save(filter_weights, 5)
    


  

def fft_weights_and_save(filter_weights, layer):
  # Pad filters right and down
    padding = 0
    if (layer == 1):
        padding = 150 - filter_weights.shape[0]
    elif (layer == 2):
        padding = 75 - filter_weights.shape[0]
    elif (layer == 3):
        padding = 37 - filter_weights.shape[0]
    elif (layer == 4):
        padding = 18 - filter_weights.shape[0]
    elif (layer == 5):
        padding = 9 - filter_weights.shape[0]

    paddings = tf.constant([[0, padding], [0, padding], [0, 0], [0, 0]])
    filter_weights = tf.pad(filter_weights, paddings, "CONSTANT", constant_values=0)

    # Transpose filters to have H,W as last 2 indices
    # H,W,C_in,C_out -> C_in,C_out,H,W
    filter_weights = tf.transpose(filter_weights, [2, 3, 0, 1])
    # FFT filters and split real and imag
    filter_weights = tf.cast(filter_weights, tf.complex64)
    filter_weights = tf.signal.fft2d(filter_weights)
    filters_real = tf.math.real(filter_weights)
    filters_imag = tf.math.imag(filter_weights)


    # Transpose filters C_in,C_out,H,W -> H,W,C_in,C_out
    filters_real = tf.transpose(filters_real, [2,3,0,1])
    filters_imag = tf.transpose(filters_imag, [2,3,0,1])

    # To numpy arrays
    filters_real = np.array(filters_real)
    filters_imag = np.array(filters_imag)

    # Save numpy arrays
    filename_real = "conv_weights_real_layer" + str(layer)
    filename_imag = "conv_weights_imag_layer" + str(layer)
    path = "C_Implementation/weight_files/"

    np.save(path + filename_real + ".npy", filters_real)
    np.save(path + filename_imag + ".npy", filters_imag)

    filters_shape = filters_real.shape

    out_real_txt = ""
    out_imag_txt = ""

    for i in range(filters_shape[0]):
        for j in range(filters_shape[1]):
            for k in range(filters_shape[2]):
                for l in range(filters_shape[3]):
                    out_real_txt += str("%10.7f" % filters_real[i][j][k][l]) + " "
                    out_imag_txt += str("%10.7f" % filters_imag[i][j][k][l]) + " "
                out_real_txt += "\n"
                out_imag_txt += "\n"


    with open(path + filename_real + ".txt", "w") as f:
        print(out_real_txt, file=f)

    with open(path + filename_imag + ".txt", "w") as f:
        print(out_imag_txt, file=f)


def b_norm_layer_weights():

    filenames = [
        {
            "real_gamma": "fourier_model-fourier_conv_layer-batch_normalization-gamma_0.npy",
            "imag_gamma": "fourier_model-fourier_conv_layer-batch_normalization_1-gamma_0.npy",
            "real_beta": "fourier_model-fourier_conv_layer-batch_normalization-beta_0.npy",
            "imag_beta": "fourier_model-fourier_conv_layer-batch_normalization_1-beta_0.npy",
            "real_moving_mean": "fourier_model-fourier_conv_layer-batch_normalization-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-fourier_conv_layer-batch_normalization_1-moving_mean_0.npy",
            "real_moving_var": "fourier_model-fourier_conv_layer-batch_normalization-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-fourier_conv_layer-batch_normalization_1-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-fourier_conv_layer_1-batch_normalization_2-gamma_0.npy",
            "imag_gamma": "fourier_model-fourier_conv_layer_1-batch_normalization_3-gamma_0.npy",
            "real_beta": "fourier_model-fourier_conv_layer_1-batch_normalization_2-beta_0.npy",
            "imag_beta": "fourier_model-fourier_conv_layer_1-batch_normalization_3-beta_0.npy",
            "real_moving_mean": "fourier_model-fourier_conv_layer_1-batch_normalization_2-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-fourier_conv_layer_1-batch_normalization_3-moving_mean_0.npy",
            "real_moving_var": "fourier_model-fourier_conv_layer_1-batch_normalization_2-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-fourier_conv_layer_1-batch_normalization_3-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-fourier_conv_layer_2-batch_normalization_4-gamma_0.npy",
            "imag_gamma": "fourier_model-fourier_conv_layer_2-batch_normalization_5-gamma_0.npy",
            "real_beta": "fourier_model-fourier_conv_layer_2-batch_normalization_4-beta_0.npy",
            "imag_beta": "fourier_model-fourier_conv_layer_2-batch_normalization_5-beta_0.npy",
            "real_moving_mean": "fourier_model-fourier_conv_layer_2-batch_normalization_4-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-fourier_conv_layer_2-batch_normalization_5-moving_mean_0.npy",
            "real_moving_var": "fourier_model-fourier_conv_layer_2-batch_normalization_4-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-fourier_conv_layer_2-batch_normalization_5-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-fourier_conv_layer_3-batch_normalization_6-gamma_0.npy",
            "imag_gamma": "fourier_model-fourier_conv_layer_3-batch_normalization_7-gamma_0.npy",
            "real_beta": "fourier_model-fourier_conv_layer_3-batch_normalization_6-beta_0.npy",
            "imag_beta": "fourier_model-fourier_conv_layer_3-batch_normalization_7-beta_0.npy",
            "real_moving_mean": "fourier_model-fourier_conv_layer_3-batch_normalization_6-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-fourier_conv_layer_3-batch_normalization_7-moving_mean_0.npy",
            "real_moving_var": "fourier_model-fourier_conv_layer_3-batch_normalization_6-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-fourier_conv_layer_3-batch_normalization_7-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-fourier_conv_layer_4-batch_normalization_8-gamma_0.npy",
            "imag_gamma": "fourier_model-fourier_conv_layer_4-batch_normalization_9-gamma_0.npy",
            "real_beta": "fourier_model-fourier_conv_layer_4-batch_normalization_8-beta_0.npy",
            "imag_beta": "fourier_model-fourier_conv_layer_4-batch_normalization_9-beta_0.npy",
            "real_moving_mean": "fourier_model-fourier_conv_layer_4-batch_normalization_8-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-fourier_conv_layer_4-batch_normalization_9-moving_mean_0.npy",
            "real_moving_var": "fourier_model-fourier_conv_layer_4-batch_normalization_8-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-fourier_conv_layer_4-batch_normalization_9-moving_variance_0.npy",
        },
        # Dense Layers:
        {
            "real_gamma": "fourier_model-complex_dense_layer-batch_normalization_10-gamma_0.npy",
            "imag_gamma": "fourier_model-complex_dense_layer-batch_normalization_11-gamma_0.npy",
            "real_beta": "fourier_model-complex_dense_layer-batch_normalization_10-beta_0.npy",
            "imag_beta": "fourier_model-complex_dense_layer-batch_normalization_11-beta_0.npy",
            "real_moving_mean": "fourier_model-complex_dense_layer-batch_normalization_10-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-complex_dense_layer-batch_normalization_11-moving_mean_0.npy",
            "real_moving_var": "fourier_model-complex_dense_layer-batch_normalization_10-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-complex_dense_layer-batch_normalization_11-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-complex_dense_layer_1-batch_normalization_12-gamma_0.npy",
            "imag_gamma": "fourier_model-complex_dense_layer_1-batch_normalization_13-gamma_0.npy",
            "real_beta": "fourier_model-complex_dense_layer_1-batch_normalization_12-beta_0.npy",
            "imag_beta": "fourier_model-complex_dense_layer_1-batch_normalization_13-beta_0.npy",
            "real_moving_mean": "fourier_model-complex_dense_layer_1-batch_normalization_12-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-complex_dense_layer_1-batch_normalization_13-moving_mean_0.npy",
            "real_moving_var": "fourier_model-complex_dense_layer_1-batch_normalization_12-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-complex_dense_layer_1-batch_normalization_13-moving_variance_0.npy",
        },
        {
            "real_gamma": "fourier_model-complex_dense_layer_2-batch_normalization_14-gamma_0.npy",
            "imag_gamma": "fourier_model-complex_dense_layer_2-batch_normalization_15-gamma_0.npy",
            "real_beta": "fourier_model-complex_dense_layer_2-batch_normalization_14-beta_0.npy",
            "imag_beta": "fourier_model-complex_dense_layer_2-batch_normalization_15-beta_0.npy",
            "real_moving_mean": "fourier_model-complex_dense_layer_2-batch_normalization_14-moving_mean_0.npy",
            "imag_moving_mean": "fourier_model-complex_dense_layer_2-batch_normalization_15-moving_mean_0.npy",
            "real_moving_var": "fourier_model-complex_dense_layer_2-batch_normalization_14-moving_variance_0.npy",
            "imag_moving_var": "fourier_model-complex_dense_layer_2-batch_normalization_15-moving_variance_0.npy",
        },


    ]
    
    
    filepath = "C_Implementation/trained_weights/"
    save_path = "C_Implementation/weight_files/"

    for i in range(len(filenames)):

        for key in filenames[i]:
            weights = np.load(filepath+filenames[i][key])
            np.save(save_path+"b_norm"+str(i+1)+"_"+key+".npy", weights)

            text_output = ""
            for j in range(len(weights)):
                text_output += str("%10.7f" % weights[j]) + " "

            with open(save_path+"b_norm"+str(i+1)+"_"+key+".txt", "w") as f:
                print(text_output, file=f)



def dense_layer_weights():

    filenames = [
        {
            "real_kernel": "fourier_model-complex_dense_layer-dense-kernel_0.npy",
            "real_bias": "fourier_model-complex_dense_layer-dense-bias_0.npy",
            "imag_kernel": "fourier_model-complex_dense_layer-dense_1-kernel_0.npy",
            "imag_bias": "fourier_model-complex_dense_layer-dense_1-bias_0.npy"
        },
        {
            "real_kernel": "fourier_model-complex_dense_layer_1-dense_2-kernel_0.npy",
            "real_bias": "fourier_model-complex_dense_layer_1-dense_2-bias_0.npy",
            "imag_kernel": "fourier_model-complex_dense_layer_1-dense_3-kernel_0.npy",
            "imag_bias": "fourier_model-complex_dense_layer_1-dense_3-bias_0.npy"
        },
        {
            "real_kernel": "fourier_model-complex_dense_layer_2-dense_4-kernel_0.npy",
            "real_bias": "fourier_model-complex_dense_layer_2-dense_4-bias_0.npy",
            "imag_kernel": "fourier_model-complex_dense_layer_2-dense_5-kernel_0.npy",
            "imag_bias": "fourier_model-complex_dense_layer_2-dense_5-bias_0.npy"
        },
        {
            "kernel": "fourier_model-dense_6-kernel_0.npy",
            "bias": "fourier_model-dense_6-bias_0.npy"
        }
    ]

    filepath = "C_Implementation/trained_weights/"
    save_path = "C_Implementation/weight_files/"

    for i in range(len(filenames)):
        for key in filenames[i]:
            weights = np.load(filepath+filenames[i][key])
            np.save(save_path+"dense"+str(i+1)+"_"+key+".npy", weights)


            text_output = ""
            if weights.ndim == 1:
                for element in weights:
                    text_output += str("%10.7f" % element) + " "
            elif weights.ndim == 2:
                for row in weights:
                    for column in row:
                        text_output += str("%10.7f" % column) + " "
                    text_output += "\n"

            with open(save_path+"dense"+str(i+1)+"_"+key+".txt", "w") as f:
                print(text_output, file=f)



        




if __name__ == "__main__":
    conv_layer_weights()
    b_norm_layer_weights()
    dense_layer_weights()