#include <iostream>
#include <fstream>
#include <cmath>
#include "layers.h"



int main() {

    // Feature maps
    FeatureMap fMap;
    // File Pointers
    FILE *realFp = fopen("test_images/x_test_real.txt", "r"); // real images
    FILE *imagFp = fopen("test_images/x_test_imag.txt", "r"); // imag images
    // Read input feature maps
    readInput(&realFp, &imagFp, &fMap);


    // Frequency Domain Filters
    FilterFreq filtersConv1, filtersConv2, filtersConv3, filtersConv4, filtersConv5;

    /// Filter File Pointers
    // Layer 1
    FILE *filtersRealL1Fp = fopen("weight_files/conv_weights_real_layer1.txt", "r"); // real filters
    FILE *filtersImagL1Fp = fopen("weight_files/conv_weights_imag_layer1.txt", "r"); // imag filters
    // Layer 2
    FILE *filtersRealL2Fp = fopen("weight_files/conv_weights_real_layer2.txt", "r"); // real filters
    FILE *filtersImagL2Fp = fopen("weight_files/conv_weights_imag_layer2.txt", "r"); // imag filters
    // Layer 3
    FILE *filtersRealL3Fp = fopen("weight_files/conv_weights_real_layer3.txt", "r"); // real filters
    FILE *filtersImagL3Fp = fopen("weight_files/conv_weights_imag_layer3.txt", "r"); // imag filters
    // Layer 4
    FILE *filtersRealL4Fp = fopen("weight_files/conv_weights_real_layer4.txt", "r"); // real filters
    FILE *filtersImagL4Fp = fopen("weight_files/conv_weights_imag_layer4.txt", "r"); // imag filters
    // Layer 5
    FILE *filtersRealL5Fp = fopen("weight_files/conv_weights_real_layer5.txt", "r"); // real filters
    FILE *filtersImagL5Fp = fopen("weight_files/conv_weights_imag_layer5.txt", "r"); // imag filters

    // Read Frequency Domain Filters
    readFilters(&filtersRealL1Fp, &filtersImagL1Fp, &filtersConv1, 1);
    readFilters(&filtersRealL2Fp, &filtersImagL2Fp, &filtersConv2, 2);
    readFilters(&filtersRealL3Fp, &filtersImagL3Fp, &filtersConv3, 3);
    readFilters(&filtersRealL4Fp, &filtersImagL4Fp, &filtersConv4, 4);
    readFilters(&filtersRealL5Fp, &filtersImagL5Fp, &filtersConv5, 5);

    // Batch Norm Values
    BatchNormLayerValues bNormValuesL1, bNormValuesL2, bNormValuesL3, bNormValuesL4, bNormValuesL5;
    BatchNormLayerValues bNormValuesL6, bNormValuesL7, bNormValuesL8; // Dense layers bnorm values

    // Batch Norm File Pointers; Order: gamma, beta, movingMean, movingVar
    FILE *bNormRealL1Fp[4] = {
        fopen("weight_files/b_norm1_real_gamma.txt", "r"),
        fopen("weight_files/b_norm1_real_beta.txt", "r"),
        fopen("weight_files/b_norm1_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm1_real_moving_var.txt", "r")
    };
    FILE *bNormImagL1Fp[4] = {
        fopen("weight_files/b_norm1_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm1_imag_beta.txt", "r"),
        fopen("weight_files/b_norm1_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm1_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL2Fp[4] = {
        fopen("weight_files/b_norm2_real_gamma.txt", "r"),
        fopen("weight_files/b_norm2_real_beta.txt", "r"),
        fopen("weight_files/b_norm2_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm2_real_moving_var.txt", "r")
    };
    FILE *bNormImagL2Fp[4] = {
        fopen("weight_files/b_norm2_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm2_imag_beta.txt", "r"),
        fopen("weight_files/b_norm2_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm2_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL3Fp[4] = {
        fopen("weight_files/b_norm3_real_gamma.txt", "r"),
        fopen("weight_files/b_norm3_real_beta.txt", "r"),
        fopen("weight_files/b_norm3_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm3_real_moving_var.txt", "r")
    };
    FILE *bNormImagL3Fp[4] = {
        fopen("weight_files/b_norm3_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm3_imag_beta.txt", "r"),
        fopen("weight_files/b_norm3_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm3_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL4Fp[4] = {
        fopen("weight_files/b_norm4_real_gamma.txt", "r"),
        fopen("weight_files/b_norm4_real_beta.txt", "r"),
        fopen("weight_files/b_norm4_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm4_real_moving_var.txt", "r")
    };
    FILE *bNormImagL4Fp[4] = {
        fopen("weight_files/b_norm4_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm4_imag_beta.txt", "r"),
        fopen("weight_files/b_norm4_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm4_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL5Fp[4] = {
        fopen("weight_files/b_norm5_real_gamma.txt", "r"),
        fopen("weight_files/b_norm5_real_beta.txt", "r"),
        fopen("weight_files/b_norm5_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm5_real_moving_var.txt", "r")
    };
    FILE *bNormImagL5Fp[4] = {
        fopen("weight_files/b_norm5_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm5_imag_beta.txt", "r"),
        fopen("weight_files/b_norm5_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm5_imag_moving_var.txt", "r")
    };
    // Dense Layers Batch Norm:
    FILE *bNormRealL6Fp[4] = {
        fopen("weight_files/b_norm6_real_gamma.txt", "r"),
        fopen("weight_files/b_norm6_real_beta.txt", "r"),
        fopen("weight_files/b_norm6_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm6_real_moving_var.txt", "r")
    };
    FILE *bNormImagL6Fp[4] = {
        fopen("weight_files/b_norm6_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm6_imag_beta.txt", "r"),
        fopen("weight_files/b_norm6_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm6_imag_moving_var.txt", "r")
    };

    FILE *bNormRealL7Fp[4] = {
        fopen("weight_files/b_norm7_real_gamma.txt", "r"),
        fopen("weight_files/b_norm7_real_beta.txt", "r"),
        fopen("weight_files/b_norm7_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm7_real_moving_var.txt", "r")
    };
    FILE *bNormImagL7Fp[4] = {
        fopen("weight_files/b_norm7_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm7_imag_beta.txt", "r"),
        fopen("weight_files/b_norm7_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm7_imag_moving_var.txt", "r")
    };

    FILE *bNormRealL8Fp[4] = {
        fopen("weight_files/b_norm8_real_gamma.txt", "r"),
        fopen("weight_files/b_norm8_real_beta.txt", "r"),
        fopen("weight_files/b_norm8_real_moving_mean.txt", "r"),
        fopen("weight_files/b_norm8_real_moving_var.txt", "r")
    };
    FILE *bNormImagL8Fp[4] = {
        fopen("weight_files/b_norm8_imag_gamma.txt", "r"),
        fopen("weight_files/b_norm8_imag_beta.txt", "r"),
        fopen("weight_files/b_norm8_imag_moving_mean.txt", "r"),
        fopen("weight_files/b_norm8_imag_moving_var.txt", "r")
    };


    // Read Batch Norm Layer Values
    readBatchNormLayerValues(bNormRealL1Fp, bNormImagL1Fp, &bNormValuesL1, CONV1_C_OUT);
    readBatchNormLayerValues(bNormRealL2Fp, bNormImagL2Fp, &bNormValuesL2, CONV2_C_OUT);
    readBatchNormLayerValues(bNormRealL3Fp, bNormImagL3Fp, &bNormValuesL3, CONV3_C_OUT);
    readBatchNormLayerValues(bNormRealL4Fp, bNormImagL4Fp, &bNormValuesL4, CONV4_C_OUT);
    readBatchNormLayerValues(bNormRealL5Fp, bNormImagL5Fp, &bNormValuesL5, CONV5_C_OUT);
    // Read Dense Layer Bnorms
    readBatchNormLayerValues(bNormRealL6Fp, bNormImagL6Fp, &bNormValuesL6, FC1_NEURONS);
    readBatchNormLayerValues(bNormRealL7Fp, bNormImagL7Fp, &bNormValuesL7, FC2_NEURONS);
    readBatchNormLayerValues(bNormRealL8Fp, bNormImagL8Fp, &bNormValuesL8, FC3_NEURONS);



    // Dense Layer Values
    DenseLayerValues denseValuesL1, denseValuesL2, denseValuesL3;

    // Dense Layer file pointers; Order: Kernels, Bias
    FILE *denseRealL1Fp[2] {
        fopen("weight_files/dense1_real_kernel.txt", "r"),
        fopen("weight_files/dense1_real_bias.txt", "r")
    };
    FILE *denseImagL1Fp[2] {
        fopen("weight_files/dense1_imag_kernel.txt", "r"),
        fopen("weight_files/dense1_imag_bias.txt", "r")
    };

    FILE *denseRealL2Fp[2] {
        fopen("weight_files/dense2_real_kernel.txt", "r"),
        fopen("weight_files/dense2_real_bias.txt", "r")
    };
    FILE *denseImagL2Fp[2] {
        fopen("weight_files/dense2_imag_kernel.txt", "r"),
        fopen("weight_files/dense2_imag_bias.txt", "r")
    };

    FILE *denseRealL3Fp[2] {
        fopen("weight_files/dense3_real_kernel.txt", "r"),
        fopen("weight_files/dense3_real_bias.txt", "r")
    };
    FILE *denseImagL3Fp[2] {
        fopen("weight_files/dense3_imag_kernel.txt", "r"),
        fopen("weight_files/dense3_imag_bias.txt", "r")
    };

    // Read Dense Layer Values
    readDenseLayerValues(denseRealL1Fp, denseImagL1Fp, &denseValuesL1, 1);
    readDenseLayerValues(denseRealL2Fp, denseImagL2Fp, &denseValuesL2, 2);
    readDenseLayerValues(denseRealL3Fp, denseImagL3Fp, &denseValuesL3, 3);

    // Output Layer Values
    OutputLayerValues outputLayerValues;

    // Output Layer file pointers, Order: Kernel, Bias
    FILE *outputDenseFp[2] {
        fopen("weight_files/dense4_kernel.txt", "r"),
        fopen("weight_files/dense4_bias.txt", "r")
    };

    // Read Output Layer Values
    readOutputLayerValues(outputDenseFp, &outputLayerValues);








    /// ---------------- Actual Network --------------- /// 
    /// BLOCK 1:
    // Fourier Conv 1
    elementWiseMultiply(&fMap, &filtersConv1);
    // Batch Normalize 1
    batchNormLayer(&fMap, &bNormValuesL1); 
    // Leaky ReLu 1
    leakyReLuLayer(&fMap);
    // Complex Pool 1
    maxPool2D(&fMap);

    /// BLOCK 2:
    // Fourier Conv 
    elementWiseMultiply(&fMap, &filtersConv2);
    // Batch Normalize 
    batchNormLayer(&fMap, &bNormValuesL2);
    // Leaky ReLu 
    leakyReLuLayer(&fMap);
    // Complex Pool
    maxPool2D(&fMap);

    /// BLOCK 3:
    // Fourier Conv 
    elementWiseMultiply(&fMap, &filtersConv3);
    // Batch Normalize 
    batchNormLayer(&fMap, &bNormValuesL3);
    // Leaky ReLu 
    leakyReLuLayer(&fMap);
    // Complex Pool 
    maxPool2D(&fMap);

    /// BLOCK 4:
    // Fourier Conv 
    elementWiseMultiply(&fMap, &filtersConv4);
    // Batch Normalize 
    batchNormLayer(&fMap, &bNormValuesL4);
    // Leaky ReLu 
    leakyReLuLayer(&fMap);
    // Complex Pool
    maxPool2D(&fMap);

    /// BLOCK 5:
    // Fourier Conv 
    elementWiseMultiply(&fMap, &filtersConv5);
    // Batch Normalize 
    batchNormLayer(&fMap, &bNormValuesL5);
    // Leaky ReLu 
    leakyReLuLayer(&fMap);
    // Complex Pool
    maxPool2D(&fMap);

    /// FLATTEN

    /// FC1
    // Dense
    denseLayer(&fMap, &denseValuesL1);
    // Batch Normalize
    batchNormLayer(&fMap, &bNormValuesL6);
    // Leaky ReLu
    leakyReLuLayer(&fMap);


    /// FC2
    // Dense
    denseLayer(&fMap, &denseValuesL2);
    // Batch Normalize
    batchNormLayer(&fMap, &bNormValuesL7);
    // Leaky ReLu
    leakyReLuLayer(&fMap);

    /// FC3
    // Dense
    denseLayer(&fMap, &denseValuesL3);
    // Batch Normalize
    batchNormLayer(&fMap, &bNormValuesL8);
    // Leaky ReLu
    leakyReLuLayer(&fMap);

    /// OUTPUT
    float *predictions;
    outputLayer(&fMap, &outputLayerValues, &predictions);



    // For testing
    float arr[3] = {0};
    for (int i = 0; i < 3; i++) {
        arr[i] = *(predictions+ i);
    }


    return 0;
}