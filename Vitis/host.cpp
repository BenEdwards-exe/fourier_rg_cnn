
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <cstring>

// C-FCNN
#include "deprecated/xrt.h"
#include "layers.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


int inSizes[5][3] = {
        {CONV1_H, CONV1_W, CONV1_C_IN},
        {CONV2_H, CONV2_W, CONV2_C_IN},
        {CONV3_H, CONV3_W, CONV3_C_IN},
        {CONV4_H, CONV4_W, CONV4_C_IN},
        {CONV5_H, CONV5_W, CONV5_C_IN},
    }; 
int outSizes[5][3] = {
    {CONV1_H, CONV1_W, CONV1_C_OUT},
    {CONV2_H, CONV2_W, CONV2_C_OUT},
    {CONV3_H, CONV3_W, CONV3_C_OUT},
    {CONV4_H, CONV4_W, CONV4_C_OUT},
    {CONV5_H, CONV5_W, CONV5_C_OUT},
}; 


// MAX KERNELS ARE 10
#define N_KERNELS INPUT_B



static void populateDeviceFilters(
    xrt::device device,
    xrt::kernel elementwiseKernels[N_KERNELS],
    xrt::bo filterReal_device[N_KERNELS],
    xrt::bo filterImag_device[N_KERNELS],
    FiltersFreq* layerFilters
) {

    int height = layerFilters->H;
    int width = layerFilters->W;
    int channelIn = layerFilters->cIn;
    int channelOut = layerFilters->cOut;

    int filterBufferSize = height * width * channelIn * channelOut;
    int filterBufferSize_bytes = filterBufferSize * sizeof(float);

    for (int i = 0; i < N_KERNELS; i++) {
        // Map device buffers
        filterReal_device[i] = xrt::bo(device, filterBufferSize_bytes, elementwiseKernels[i].group_id(4));
        filterImag_device[i] = xrt::bo(device, filterBufferSize_bytes, elementwiseKernels[i].group_id(5));
        // Write host buffers to device buffers
        filterReal_device[i].write(layerFilters->realFilterPtr);
        filterImag_device[i].write(layerFilters->imagFilterPtr);
        // Sync device buffers to device
        filterReal_device[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        filterImag_device[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

}

// Copy the Real and Imag Fmaps of a layer to the device
static void copyFmapsToDevice(
    FeatureMap* fMap,
    xrt::bo fMapsInReal_device[N_KERNELS],
    xrt::bo fMapsInImag_device[N_KERNELS]
) {
    // assert (fMap->B == 10);
    // std::cout << "Copy Feature Maps To Device\n";

    for (int i = 0; i < N_KERNELS; i++) {
        int fMapOffset = i * (fMap->H * fMap->W * fMap->C);
        fMapsInReal_device[i].write(fMap->realValPtr + fMapOffset);
        fMapsInImag_device[i].write(fMap->imagValPtr + fMapOffset);
    }

    for (int i = 0; i < N_KERNELS; i++) {
        fMapsInReal_device[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fMapsInImag_device[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

}

// Copy the Real and Imag Fmaps of a layer from the device to the host
static void copyFmapsFromDevice(
    FeatureMap* fMap,
    xrt::bo fMapsOutReal_device[N_KERNELS],
    xrt::bo fMapsOutImag_device[N_KERNELS],
    int fMapSize,
    int channelOutSize
) {

    int fMapOutSize = fMapSize * fMapSize * channelOutSize;

    float* outReal = (float*) calloc(fMapOutSize*N_KERNELS, sizeof(float));
    float* outImag = (float*) calloc(fMapOutSize*N_KERNELS, sizeof(float));

    for (int i = 0; i < N_KERNELS; i++) {
        fMapsOutReal_device[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        fMapsOutImag_device[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    }

    for (int i = 0; i < N_KERNELS; i++) {
        int ptrOffset = fMapOutSize * i;

        fMapsOutReal_device[i].read(outReal + ptrOffset);
        fMapsOutImag_device[i].read(outImag + ptrOffset);
    }


    free(fMap->realValPtr);
    free(fMap->imagValPtr);

    fMap->realValPtr = outReal;
    fMap->imagValPtr = outImag;

    fMap->C = channelOutSize;


    return;
}


// Weights should allready have been copied to the fpga
// Fmap Buffers should already exist in Device Global Memory
static void runKernelsParallel(
        xrt::kernel elwise_krnl[N_KERNELS],
        xrt::bo fMapInReal_device[N_KERNELS], xrt::bo fMapInImag_device[N_KERNELS],
        xrt::bo fMapOutReal_device[N_KERNELS], xrt::bo fMapOutImag_device[N_KERNELS],
        xrt::bo filterReal_device[N_KERNELS], xrt::bo filterImag_device[N_KERNELS],
        int fMapSize, int channelInSize, int channelOutSize
    ) {

    xrt::run krnl_run[N_KERNELS];
    for (int i = 0; i < N_KERNELS; i++) {
        krnl_run[i] = xrt::run(elwise_krnl[i]);
    }

    for (int i = 0; i < N_KERNELS; i++) {
        krnl_run[i].set_arg(0, fMapInReal_device[i]);
        krnl_run[i].set_arg(1, fMapInImag_device[i]);
        krnl_run[i].set_arg(2, fMapOutReal_device[i]);
        krnl_run[i].set_arg(3, fMapOutImag_device[i]);
        krnl_run[i].set_arg(4, filterReal_device[i]);
        krnl_run[i].set_arg(5, filterImag_device[i]);
        krnl_run[i].set_arg(6, fMapSize);
        krnl_run[i].set_arg(7, channelInSize);
        krnl_run[i].set_arg(8, channelOutSize);
    }


    for (int i = 0; i < N_KERNELS; i ++) {
        krnl_run[i].start();
    }

    for (int i = 0; i < N_KERNELS; i++) {
        krnl_run[i].wait();
    }
}




int main(int argc, char** argv) {

    if (argc < 2) {
        return EXIT_FAILURE;
    }

    // Read settings
    int device_index = 0;
    std::string hostDir = argv[0];
    std::string binaryFile = argv[1];

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Host directory " << hostDir << std::endl;
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);


    
    
    // File Pointers
    FILE *realFp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/test_images/x_test_real.txt", "r"); // real images
    FILE *imagFp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/test_images/x_test_imag.txt", "r"); // imag images
    // // Read input feature maps
    // readInput(&realFp, &imagFp, &fMap);


    // Frequency Domain Filters
    FilterFreq filtersConv1, filtersConv2, filtersConv3, filtersConv4, filtersConv5;

    /// Filter File Pointers
    // Layer 1
    FILE *filtersRealL1Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_real_layer1.txt", "r"); // real filters
    FILE *filtersImagL1Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_imag_layer1.txt", "r"); // imag filters
    // Layer 2
    FILE *filtersRealL2Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_real_layer2.txt", "r"); // real filters
    FILE *filtersImagL2Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_imag_layer2.txt", "r"); // imag filters
    // Layer 3
    FILE *filtersRealL3Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_real_layer3.txt", "r"); // real filters
    FILE *filtersImagL3Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_imag_layer3.txt", "r"); // imag filters
    // Layer 4
    FILE *filtersRealL4Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_real_layer4.txt", "r"); // real filters
    FILE *filtersImagL4Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_imag_layer4.txt", "r"); // imag filters
    // Layer 5
    FILE *filtersRealL5Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_real_layer5.txt", "r"); // real filters
    FILE *filtersImagL5Fp = fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/conv_weights_imag_layer5.txt", "r"); // imag filters

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
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_real_moving_var.txt", "r")
    };
    FILE *bNormImagL1Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm1_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL2Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_real_moving_var.txt", "r")
    };
    FILE *bNormImagL2Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm2_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL3Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_real_moving_var.txt", "r")
    };
    FILE *bNormImagL3Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm3_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL4Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_real_moving_var.txt", "r")
    };
    FILE *bNormImagL4Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm4_imag_moving_var.txt", "r")
    };
    FILE *bNormRealL5Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_real_moving_var.txt", "r")
    };
    FILE *bNormImagL5Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm5_imag_moving_var.txt", "r")
    };
    // Dense Layers Batch Norm:
    FILE *bNormRealL6Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_real_moving_var.txt", "r")
    };
    FILE *bNormImagL6Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm6_imag_moving_var.txt", "r")
    };

    FILE *bNormRealL7Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_real_moving_var.txt", "r")
    };
    FILE *bNormImagL7Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm7_imag_moving_var.txt", "r")
    };

    FILE *bNormRealL8Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_real_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_real_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_real_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_real_moving_var.txt", "r")
    };
    FILE *bNormImagL8Fp[4] = {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_imag_gamma.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_imag_beta.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_imag_moving_mean.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/b_norm8_imag_moving_var.txt", "r")
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
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense1_real_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense1_real_bias.txt", "r")
    };
    FILE *denseImagL1Fp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense1_imag_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense1_imag_bias.txt", "r")
    };

    FILE *denseRealL2Fp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense2_real_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense2_real_bias.txt", "r")
    };
    FILE *denseImagL2Fp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense2_imag_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense2_imag_bias.txt", "r")
    };

    FILE *denseRealL3Fp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense3_real_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense3_real_bias.txt", "r")
    };
    FILE *denseImagL3Fp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense3_imag_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense3_imag_bias.txt", "r")
    };

    // Read Dense Layer Values
    readDenseLayerValues(denseRealL1Fp, denseImagL1Fp, &denseValuesL1, 1);
    readDenseLayerValues(denseRealL2Fp, denseImagL2Fp, &denseValuesL2, 2);
    readDenseLayerValues(denseRealL3Fp, denseImagL3Fp, &denseValuesL3, 3);

    // Output Layer Values
    OutputLayerValues outputLayerValues;

    // Output Layer file pointers, Order: Kernel, Bias
    FILE *outputDenseFp[2] {
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense4_kernel.txt", "r"),
        fopen("/home/ben/Repos/fourier_rg_cnn/C_Implementation/weight_files/dense4_bias.txt", "r")
    };

    // Read Output Layer Values
    readOutputLayerValues(outputDenseFp, &outputLayerValues);

    
    xrt::kernel elwise_krnl[10];
    elwise_krnl[0] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_1}");
    elwise_krnl[1] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_2}");
    elwise_krnl[2] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_3}");
    elwise_krnl[3] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_4}");
    elwise_krnl[4] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_5}");
    elwise_krnl[5] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_6}");
    elwise_krnl[6] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_7}");
    elwise_krnl[7] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_8}");
    elwise_krnl[8] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_9}");
    elwise_krnl[9] = xrt::kernel(device, uuid, "elmnt_stream_krnl:{elmnt_stream_krnl_10}");



    xrt::bo filterReal_1_device[N_KERNELS], filterImag_1_device[N_KERNELS];
    xrt::bo filterReal_2_device[N_KERNELS], filterImag_2_device[N_KERNELS];
    xrt::bo filterReal_3_device[N_KERNELS], filterImag_3_device[N_KERNELS];
    xrt::bo filterReal_4_device[N_KERNELS], filterImag_4_device[N_KERNELS];
    xrt::bo filterReal_5_device[N_KERNELS], filterImag_5_device[N_KERNELS];

    populateDeviceFilters(device, elwise_krnl, filterReal_1_device, filterImag_1_device, &filtersConv1);
    populateDeviceFilters(device, elwise_krnl, filterReal_2_device, filterImag_2_device, &filtersConv2);
    populateDeviceFilters(device, elwise_krnl, filterReal_3_device, filterImag_3_device, &filtersConv3);
    populateDeviceFilters(device, elwise_krnl, filterReal_4_device, filterImag_4_device, &filtersConv4);
    populateDeviceFilters(device, elwise_krnl, filterReal_5_device, filterImag_5_device, &filtersConv5);


    std::cout << "Allocate Feature Map In and Out Buffers on for Device\n";
    // Buffers to hold the input feature maps for each layer and each compute unit
    xrt::bo fMapsInReal_device[5][N_KERNELS], fMapsInImag_device[5][N_KERNELS];
    xrt::bo fMapsOutReal_device[5][N_KERNELS], fMapsOutImag_device[5][N_KERNELS];

    for (int layer = 0; layer < 5; layer++) {
        for (int cu = 0; cu < N_KERNELS; cu++) {
            float fMapInBytes = inSizes[layer][0] * inSizes[layer][1] * inSizes[layer][2] * sizeof(float);
            float fMapOutBytes = outSizes[layer][0] * outSizes[layer][1] * outSizes[layer][2] * sizeof(float);
            fMapsInReal_device[layer][cu] = xrt::bo(device, fMapInBytes, elwise_krnl[cu].group_id(0));
            fMapsInImag_device[layer][cu] = xrt::bo(device, fMapInBytes, elwise_krnl[cu].group_id(1));
            fMapsOutReal_device[layer][cu] = xrt::bo(device, fMapOutBytes, elwise_krnl[cu].group_id(2));
            fMapsOutImag_device[layer][cu] = xrt::bo(device, fMapOutBytes, elwise_krnl[cu].group_id(3));
        }
    }

    

    auto network_start = std::chrono::high_resolution_clock::now();



    int total_images = 120;
    int runs = total_images / N_KERNELS;

    for (int i = 0; i < runs; i++) {
        // Create FMAP
        FeatureMap fMap;

        // Read input feature maps
        readInput(&realFp, &imagFp, &fMap);

        /// ---------------- Actual Network --------------- /// 
        /// BLOCK 1:
        // Fourier Conv 1
        copyFmapsToDevice(&fMap, fMapsInReal_device[0], fMapsInImag_device[0]);
        runKernelsParallel(elwise_krnl, fMapsInReal_device[0], fMapsInImag_device[0], fMapsOutReal_device[0], fMapsOutImag_device[0], filterReal_1_device, filterImag_1_device, CONV1_H, CONV1_C_IN, CONV1_C_OUT);
        copyFmapsFromDevice(&fMap, fMapsOutReal_device[0], fMapsOutImag_device[0], outSizes[0][0], outSizes[0][2]);
        // elementWiseMultiply(&fMap, &filtersConv1);
        // Batch Normalize 1
        batchNormLayer(&fMap, &bNormValuesL1); 
        // Leaky ReLu 1
        leakyReLuLayer(&fMap);
        // Complex Pool 1
        maxPool2D(&fMap);

        /// BLOCK 2:
        // Fourier Conv 
        copyFmapsToDevice(&fMap, fMapsInReal_device[1], fMapsInImag_device[1]);
        runKernelsParallel(elwise_krnl, fMapsInReal_device[1], fMapsInImag_device[1], fMapsOutReal_device[1], fMapsOutImag_device[1], filterReal_2_device, filterImag_2_device, CONV2_H, CONV2_C_IN, CONV2_C_OUT);
        copyFmapsFromDevice(&fMap, fMapsOutReal_device[1], fMapsOutImag_device[1], outSizes[1][0], outSizes[1][2]);
        // elementWiseMultiply(&fMap, &filtersConv2);
        // Batch Normalize 
        batchNormLayer(&fMap, &bNormValuesL2);
        // Leaky ReLu 
        leakyReLuLayer(&fMap);
        // Complex Pool
        maxPool2D(&fMap);

        /// BLOCK 3:
        // Fourier Conv 
        copyFmapsToDevice(&fMap, fMapsInReal_device[2], fMapsInImag_device[2]);
        runKernelsParallel(elwise_krnl, fMapsInReal_device[2], fMapsInImag_device[2], fMapsOutReal_device[2], fMapsOutImag_device[2], filterReal_3_device, filterImag_3_device, CONV3_H, CONV3_C_IN, CONV3_C_OUT);
        copyFmapsFromDevice(&fMap, fMapsOutReal_device[2], fMapsOutImag_device[2], outSizes[2][0], outSizes[2][2]);
        // elementWiseMultiply(&fMap, &filtersConv3);
        // Batch Normalize 
        batchNormLayer(&fMap, &bNormValuesL3);
        // Leaky ReLu 
        leakyReLuLayer(&fMap);
        // Complex Pool 
        maxPool2D(&fMap);

        /// BLOCK 4:
        // Fourier Conv 
        copyFmapsToDevice(&fMap, fMapsInReal_device[3], fMapsInImag_device[3]);
        runKernelsParallel(elwise_krnl, fMapsInReal_device[3], fMapsInImag_device[3], fMapsOutReal_device[3], fMapsOutImag_device[3], filterReal_4_device, filterImag_4_device, CONV4_H, CONV4_C_IN, CONV4_C_OUT);
        copyFmapsFromDevice(&fMap, fMapsOutReal_device[3], fMapsOutImag_device[3], outSizes[3][0], outSizes[3][2]);
        // elementWiseMultiply(&fMap, &filtersConv4);
        // Batch Normalize 
        batchNormLayer(&fMap, &bNormValuesL4);
        // Leaky ReLu 
        leakyReLuLayer(&fMap);
        // Complex Pool
        maxPool2D(&fMap);

        /// BLOCK 5:
        // Fourier Conv 
        copyFmapsToDevice(&fMap, fMapsInReal_device[4], fMapsInImag_device[4]);
        runKernelsParallel(elwise_krnl, fMapsInReal_device[4], fMapsInImag_device[4], fMapsOutReal_device[4], fMapsOutImag_device[4], filterReal_5_device, filterImag_5_device, CONV5_H, CONV5_C_IN, CONV5_C_OUT);
        copyFmapsFromDevice(&fMap, fMapsOutReal_device[4], fMapsOutImag_device[4], outSizes[4][0], outSizes[4][2]);
        // elementWiseMultiply(&fMap, &filtersConv5);
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


        // Predictions
        float arr[3] = {0};
        for (int i = 0; i < 3; i++) {
            arr[i] = *(predictions+ i);
        }

        free(fMap.imagValPtr);
        free(fMap.realValPtr);

        // std::cout << "RUN " << i+1 << " completed\n";
    }
    
    auto network_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> network_duration = network_end - network_start;
    std::cout << "Total Images: " << total_images << "\n";
    std::cout << "Network total time TOTAL_IMAGES images " << network_duration.count() << " s\n";
    std::cout << "Network time per image " << network_duration.count()/total_images << " s\n";




    return 0;
}

