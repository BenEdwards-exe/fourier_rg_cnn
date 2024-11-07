#include <iostream>
#include <fstream>
#include <cmath>
#include "layers.h"


/// ------------------------ Functions that read in values ------------------------ ///

void readInput(FILE **realFp, FILE **imagFp, FeatureMap *fMap) {
    fMap->B = INPUT_B;
    fMap->H = INPUT_H;
    fMap->W = INPUT_W;
    fMap->C = INPUT_C;

    // Allocate memory
    fMap->realValPtr = (float *) malloc(fMap->B * fMap->H * fMap->W * fMap->C * sizeof(float));
    fMap->imagValPtr = (float *) malloc(fMap->B * fMap->H * fMap->W * fMap->C * sizeof(float));

    // (i, j, k, l) ; [D1, D2, D3, D4] ; index = l + k*D4 + j*D4*D3 + i*D4*D3*D2
    int memoryOffset = 0;
    float temp = 0;
    for (int i = 0; i < fMap->B; i++) {
        for (int j = 0; j < fMap->H; j++) {
            for (int k = 0; k < fMap->W; k++) {
                for (int l = 0; l < fMap->C; l++) {
                    // Calculate offset
                    memoryOffset = l + k*fMap->C + j*fMap->C*fMap->W + i*fMap->C*fMap->W*fMap->H;
                    // Read real part of input
                    fscanf(
                        *realFp, 
                        "%f", 
                        &temp
                    );
                    *(fMap->realValPtr + memoryOffset) = temp;
                    // Read imag part of input
                    fscanf(
                        *imagFp, 
                        "%f", 
                        &temp
                    );
                    *(fMap->imagValPtr + memoryOffset) = temp;

                } // Channels
            } // Width
        } // Height
    } // Batch

}



void readFilters(FILE **realFilterFp, FILE **imagFilterFp, FiltersFreq *filters, int layerNum) {
    switch (layerNum) {
        case 1:
            filters->H = CONV1_H;
            filters->W = CONV1_W;
            filters->cIn = CONV1_C_IN;
            filters->cOut = CONV1_C_OUT;
            break;

        case 2:
            filters->H = CONV2_H;
            filters->W = CONV2_W;
            filters->cIn = CONV2_C_IN;
            filters->cOut = CONV2_C_OUT;
            break;

        case 3:
            filters->H = CONV3_H;
            filters->W = CONV3_W;
            filters->cIn = CONV3_C_IN;
            filters->cOut = CONV3_C_OUT;
            break;
        
        case 4:
            filters->H = CONV4_H;
            filters->W = CONV4_W;
            filters->cIn = CONV4_C_IN;
            filters->cOut = CONV4_C_OUT;
            break;

        case 5:
            filters->H = CONV5_H;
            filters->W = CONV5_W;
            filters->cIn = CONV5_C_IN;
            filters->cOut = CONV5_C_OUT;
            break;

        default:
            break;
    }

    // Allocate Memory
    filters->realFilterPtr = (float*) malloc(filters->H * filters->W * filters->cIn * filters->cOut * sizeof(float));
    filters->imagFilterPtr = (float*) malloc(filters->H * filters->W * filters->cIn * filters->cOut * sizeof(float));


    // (i, j, k, l) ; [D1, D2, D3, D4] ; index = l + k*D4 + j*D4*D3 + i*D4*D3*D2
    int memoryOffset = 0;
    float temp = 0;
    for (int i = 0; i < filters->H; i++) {
        for (int j = 0; j < filters->W; j++) {
            for (int k = 0; k < filters->cIn; k++) {
                for (int l = 0; l < filters->cOut; l++) {
                    // Read real filters
                    memoryOffset = l + k*filters->cOut + j*filters->cOut*filters->cIn + i*filters->cOut*filters->cIn*filters->W;
                    fscanf(
                        *realFilterFp,
                        "%f",
                        &temp   
                    );
                    *(filters->realFilterPtr + memoryOffset) = temp;
                    // Read imag filters
                    fscanf(
                        *imagFilterFp,
                        "%f",
                        &temp
                    );
                    *(filters->imagFilterPtr + memoryOffset) = temp;

                } // Channel_Out
            } // Channel_In
        } // Width
    } // Height


}



void readBatchNormLayerValues(FILE *realFilePtrs[4], FILE *imagFilePtrs[4], BatchNormLayerValues *bNormLayerValues, int layerChannels) {
    bNormLayerValues->outputChannels = layerChannels;

    // Allocate memory
    bNormLayerValues->gammaReal = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->betaReal = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->movingMeanReal = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->movingVarReal = (float *) malloc(layerChannels * sizeof(float));

    bNormLayerValues->gammaImag = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->betaImag = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->movingMeanImag = (float *) malloc(layerChannels * sizeof(float));
    bNormLayerValues->movingVarImag = (float *) malloc(layerChannels * sizeof(float));

    // filePtrs are in order gamma, beta, movingMean, movingVar
    for (int i = 0; i < layerChannels; i++) {
        // Read real file values
        fscanf(realFilePtrs[0], "%f", (bNormLayerValues->gammaReal + i));
        fscanf(realFilePtrs[1], "%f", (bNormLayerValues->betaReal + i));
        fscanf(realFilePtrs[2], "%f", (bNormLayerValues->movingMeanReal + i));
        fscanf(realFilePtrs[3], "%f", (bNormLayerValues->movingVarReal + i));
        // Read imag file values
        fscanf(imagFilePtrs[0], "%f", (bNormLayerValues->gammaImag + i));
        fscanf(imagFilePtrs[1], "%f", (bNormLayerValues->betaImag + i));
        fscanf(imagFilePtrs[2], "%f", (bNormLayerValues->movingMeanImag + i));
        fscanf(imagFilePtrs[3], "%f", (bNormLayerValues->movingVarImag + i));
    }

    return;
}

/*
filePtrs: Array of pointers to FILE pointers; 0: Kernels, 1: Bias
denseLayerValues: Pointer to Dense Layer Values struct
*/
void readDenseLayerValues(FILE *realFilePtrs[2], FILE *imagFilePtrs[2], DenseLayerValues *denseLayerValues, int denseLayerNum) {
    switch (denseLayerNum)
    {
    case 1:
        denseLayerValues->inputNum = FC1_INPUT_NUM;
        denseLayerValues->outputNum = FC1_NEURONS;
        break;
    case 2:
        denseLayerValues->inputNum = FC1_NEURONS;
        denseLayerValues->outputNum = FC2_NEURONS;
        break;
    case 3:
        denseLayerValues->inputNum = FC2_NEURONS;
        denseLayerValues->outputNum = FC3_NEURONS;
        break;
    
    default:
        break;
    }

    // Allocate Memory
    denseLayerValues->kernelsReal = (float *) malloc(denseLayerValues->inputNum * denseLayerValues->outputNum * sizeof(float));
    denseLayerValues->kernelsImag = (float *) malloc(denseLayerValues->inputNum * denseLayerValues->outputNum * sizeof(float));
    denseLayerValues->biasReal = (float *) malloc(denseLayerValues->outputNum * sizeof(float));
    denseLayerValues->biasImag = (float *) malloc(denseLayerValues->outputNum * sizeof(float));

    // Read in kernel values
    int memoryOffset = 0;
    for (int inputIndex = 0; inputIndex < denseLayerValues->inputNum; inputIndex++) {
        for (int outputIndex = 0; outputIndex < denseLayerValues->outputNum; outputIndex++) {

            memoryOffset = outputIndex + inputIndex*denseLayerValues->outputNum;

            fscanf(realFilePtrs[0], "%f", (denseLayerValues->kernelsReal + memoryOffset));
            fscanf(imagFilePtrs[0], "%f", (denseLayerValues->kernelsImag + memoryOffset));

        } // Output Values
    } // Input Values
    
    // Read in bias values 
    for (int outputIndex = 0; outputIndex < denseLayerValues->outputNum; outputIndex++) {
        fscanf(realFilePtrs[1], "%f", (denseLayerValues->biasReal + outputIndex));
        fscanf(imagFilePtrs[1], "%f", (denseLayerValues->biasImag + outputIndex));
    }

    return;
}

/*
filePtrs: Array of pointers to FILE pointers
outputLayerValues: Pointer to Output Layer Values struct
*/
void readOutputLayerValues(FILE *outFilePtrs[2], OutputLayerValues *outputLayerValues) {
    outputLayerValues->inputNum = FC_OUT_INPUT_NUM;
    outputLayerValues->outputNum = FC_OUT_NEURONS;
    // Allocate memory
    outputLayerValues->kernelsPtr = (float*) malloc(outputLayerValues->inputNum * outputLayerValues->outputNum * sizeof(float));
    outputLayerValues->biasPtr = (float*) malloc(outputLayerValues->outputNum * sizeof(float));

    // Read in kernel values
    for (int inputIndex = 0; inputIndex < outputLayerValues->inputNum; inputIndex++) {
        for (int outputIndex = 0; outputIndex < outputLayerValues->outputNum; outputIndex++) {
            fscanf(outFilePtrs[0], "%f", (outputLayerValues->kernelsPtr + outputIndex + inputIndex*outputLayerValues->outputNum));
        }
    }
    // Read in bias values
    for (int outputIndex = 0; outputIndex < outputLayerValues->outputNum; outputIndex++) {
        fscanf(outFilePtrs[1], "%f", (outputLayerValues->biasPtr + outputIndex));
    }

    return;
}




/// ------------------------ Functions that perform layer calculations ------------ ///


// fMap shape: B,H,W,C
// filters shape: H,W,C_in,C_out
void elementWiseMultiply(FeatureMap *fMap, FilterFreq *filters) {

    // Output F-Maps
    float * outputFeaturesReal = (float*) calloc(fMap->B * fMap->H * fMap->W * filters->cOut, sizeof(float));
    float * outputFeaturesImag = (float*) calloc(fMap->B * fMap->H * fMap->W * filters->cOut, sizeof(float));
    // Use calloc to initialize to zero. This is for when we sum across Input Feature Channels

    // Temp values
    float ac, bd, ad, bc = 0;
    float featureRealCurrent, featureImagCurrent, filterRealCurrent, filterImagCurrent;
    int featureMemoryOffset = 0;
    int filterMemoryOffset = 0;
    int outputMemoryOffset = 0;
    // float * tempReal = (float*) malloc(fMap->H * fMap->W * filters->cIn * filters->cOut * sizeof(float));
    // float * tempImag = (float*) malloc(fMap->H * fMap->W * filters->cIn * filters->cOut * sizeof(float));

    // Height and Width of F-Maps and Freq-Filters are the same; C and C_In the same

    // Take one feature map in batch
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {

        // AC = Input_Real * Filter_Real
        // BD = Input_Imag * Filter_Imag
        // AD = Input_Real * Filter_Imag
        // BC = Input_Imag * Filter_Real

        for (int heightIndex = 0; heightIndex < fMap->H; heightIndex++) {
            for (int widthIndex = 0; widthIndex < fMap->W; widthIndex++) {

                // Multiply one filter channel (cOut) with all the channels of a feature map and then sum over input channels (doing it in different order though); 
                // Output channels will then be same as filter cOut
                for (int channelInIndex = 0; channelInIndex < fMap->C; channelInIndex++) {
                    // Calcualte memory offset to fmap channel and isolate current 'pixel'
                    featureMemoryOffset = channelInIndex + widthIndex*fMap->C + heightIndex*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;
                    featureRealCurrent = *(fMap->realValPtr + featureMemoryOffset);
                    featureImagCurrent = *(fMap->imagValPtr + featureMemoryOffset);

                    for (int channelOutIndex = 0; channelOutIndex < filters->cOut; channelOutIndex++) {
                        // Calculate memory offset to filter channel and isolate current 'pixel' of output channel
                        filterMemoryOffset = channelOutIndex + channelInIndex*filters->cOut + widthIndex*filters->cOut*filters->cIn + heightIndex*filters->cOut*filters->cIn*filters->W;
                        filterRealCurrent = *(filters->realFilterPtr + filterMemoryOffset);
                        filterImagCurrent = *(filters->imagFilterPtr + filterMemoryOffset);

                        ac = featureRealCurrent * filterRealCurrent;
                        bd = featureImagCurrent * filterImagCurrent;
                        ad = featureRealCurrent * filterImagCurrent;
                        bc = featureImagCurrent * filterRealCurrent;

                        // Sum across input channels and write into output memory
                        outputMemoryOffset = channelOutIndex + widthIndex*filters->cOut + heightIndex*filters->cOut*fMap->W + batchIndex*filters->cOut*fMap->W*fMap->H;
                        *(outputFeaturesReal + outputMemoryOffset) += (ac + bd);
                        *(outputFeaturesImag + outputMemoryOffset) += (bc - ad);


                    } // Output channels (i.e., filter out channel)
                } // channelInIndex loop (i.e., input feature map channels)
            } // Width loop
        } // Height loop
    } // Batch loop


    // Free Input Feature Map Memory
    free(fMap->realValPtr);
    free(fMap->imagValPtr);

    // Point fMap to output features
    fMap->realValPtr = outputFeaturesReal;
    fMap->imagValPtr = outputFeaturesImag;
    // Update channels value
    fMap->C = filters->cOut;

    return;
}

/*
fMap: Input Feature Maps (BHWC)
bNormLayerValues: Pointer to Batch Norm Layer Values struct
*/
void batchNormLayer(FeatureMap *fMap, BatchNormLayerValues *bNormLayerValues) {

    // Formula: gamma * ((featureMap - moving_mean) / sqrt(moving_var + epsilon)) + beta

    // TODO: have "sqrt(moving_var + epsilon)" saved as a value since calculation will always be the same


    // For each B,H,W,C apply Batch Norm
    int memoryOffset = 0;
    float gamma, beta, movingMean, movingVar = 0;
    float epsilon = bNormLayerValues->epsilon;
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        for (int heightIndex = 0; heightIndex < fMap->H; heightIndex++) {
            for (int widthIndex = 0; widthIndex < fMap->W; widthIndex++) {
                for (int channelIndex = 0; channelIndex < fMap->C; channelIndex++) {
                    // fMap->C and bNormLayerValues->outputChannels should be same value
                    memoryOffset = channelIndex + widthIndex*fMap->C + heightIndex*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;

                    // Real Feature Maps
                    gamma = *(bNormLayerValues->gammaReal + channelIndex);
                    beta = *(bNormLayerValues->betaReal + channelIndex);
                    movingMean = *(bNormLayerValues->movingMeanReal + channelIndex);
                    movingVar = *(bNormLayerValues->movingVarReal + channelIndex);
                    
                    *(fMap->realValPtr + memoryOffset) = gamma * ( (*(fMap->realValPtr + memoryOffset) - movingMean) / sqrt(movingVar + epsilon) ) + beta;


                    // Imag Feature Maps
                    gamma = *(bNormLayerValues->gammaImag + channelIndex);
                    beta = *(bNormLayerValues->betaImag + channelIndex);
                    movingMean = *(bNormLayerValues->movingMeanImag + channelIndex);
                    movingVar = *(bNormLayerValues->movingVarImag + channelIndex);

                    *(fMap->imagValPtr + memoryOffset) = gamma * ( (*(fMap->imagValPtr + memoryOffset) - movingMean) / sqrt(movingVar + epsilon) ) + beta;

                } // Channel
            } // Width
        } // Height
    } // Batch


    return;
}


/*
fMap: Input Feature Maps (BHWC)
Alpha value set as 0.2
*/
void leakyReLuLayer(FeatureMap *fMap) {

    int memoryOffset = 0;
    float fMapVal = 0;
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        for (int heightIndex = 0; heightIndex < fMap->H; heightIndex++) {
            for (int widthIndex = 0; widthIndex < fMap->W; widthIndex++) {
                for (int channelIndex = 0; channelIndex < fMap->C; channelIndex++) {

                    memoryOffset = channelIndex + widthIndex*fMap->C + heightIndex*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;

                    // Real feature maps
                    fMapVal = *(fMap->realValPtr + memoryOffset);
                    *(fMap->realValPtr + memoryOffset) = fmax(0.2*fMapVal, fMapVal);
                    
                    // Imag feature maps
                    fMapVal = *(fMap->imagValPtr + memoryOffset);
                    *(fMap->imagValPtr + memoryOffset) = fmax(0.2*fMapVal, fMapVal);

                } // Channel
            } // Width
        } // Height
    } // Batch

    return;
}


/*
fMap: Input Feature Maps (BHWC); Point to new memory after function
Pool size: (2,2)
Stride: (2,2)
Padding: valid
*/
void maxPool2D(FeatureMap *fMap) {

    // Height and Width of output feature maps
    int outHeight = floor((fMap->H - 2) / 2) + 1;
    int outWidth = floor((fMap->W - 2) / 2) + 1;

    // Output feature maps; Point to them at the end
    float *outputFeaturesReal = (float *) malloc(fMap->B * outHeight * outWidth * fMap->C * sizeof(float));
    float *outputFeaturesImag = (float *) malloc(fMap->B * outHeight * outWidth * fMap->C * sizeof(float));


    float fMapVal0, fMapVal1, fMapVal2, fMapVal3 = 0;
    int a, b = 0;
    int memoryOffset0, memoryOffset1, memoryOffset2, memoryOffset3 = 0;
    int outputMemoryOffset = 0;
    float fMapMax = 0;
    // Out[a][b] =  max( max(In[2a][2b],In[2a,2b+1]), max(In[2a+1][2b],In[2a+1][2b+1]) ) 
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        for (int heightIndex = 0; heightIndex < outHeight; heightIndex++) {
            for (int widthIndex = 0; widthIndex < outWidth; widthIndex++) {
                for (int channelIndex = 0; channelIndex < fMap->C; channelIndex++) {
                    
                    // Memory offset to input values
                    memoryOffset0 = channelIndex + (widthIndex*2)*fMap->C + (heightIndex*2)*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;
                    memoryOffset1 = channelIndex + (widthIndex*2 + 1)*fMap->C + (heightIndex*2)*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;
                    memoryOffset2 = channelIndex + (widthIndex*2)*fMap->C + (heightIndex*2 + 1)*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;
                    memoryOffset3 = channelIndex + (widthIndex*2 + 1)*fMap->C + (heightIndex*2 + 1)*fMap->C*fMap->W + batchIndex*fMap->C*fMap->W*fMap->H;
                    // Memory offset for output values
                    outputMemoryOffset = channelIndex + widthIndex*fMap->C + heightIndex*fMap->C*outWidth + batchIndex*fMap->C*outWidth*outHeight;

                    // Real feature maps
                    fMapVal0 = *(fMap->realValPtr + memoryOffset0);
                    fMapVal1 = *(fMap->realValPtr + memoryOffset1);
                    fMapVal2 = *(fMap->realValPtr + memoryOffset2);
                    fMapVal3 = *(fMap->realValPtr + memoryOffset3);

                    fMapMax = fmax(fmax(fMapVal0,fMapVal1), fmax(fMapVal2,fMapVal3));
                    *(outputFeaturesReal + outputMemoryOffset) = fMapMax;

                    // Imag feature maps
                    fMapVal0 = *(fMap->imagValPtr + memoryOffset0);
                    fMapVal1 = *(fMap->imagValPtr + memoryOffset1);
                    fMapVal2 = *(fMap->imagValPtr + memoryOffset2);
                    fMapVal3 = *(fMap->imagValPtr + memoryOffset3);

                    *(outputFeaturesImag + outputMemoryOffset) = fmax(fmax(fMapVal0,fMapVal1), fmax(fMapVal2,fMapVal3));

                } // Channel
            } // Out Width
        } // Out Height
    } // Batch


    // Update feature map structs
    fMap->H = outHeight;
    fMap->W = outWidth;
    free(fMap->realValPtr);
    free(fMap->imagValPtr);
    fMap->realValPtr = outputFeaturesReal;
    fMap->imagValPtr = outputFeaturesImag;


    return;
}


/*
fMap: Input Feature Maps (BHWC) [treated as "flattened", already in 1D array]
denseLayerValues: Kernels and Bias values for dense layer
*/
void denseLayer(FeatureMap *fMap, DenseLayerValues *denseLayerValues) {

    // Output feature maps; Initialize to zero with calloc; Point to them at the end
    float *outputFeaturesReal = (float *) calloc(fMap->B * denseLayerValues->outputNum, sizeof(float));
    float *outputFeaturesImag = (float *) calloc(fMap->B * denseLayerValues->outputNum, sizeof(float));


    // Loop through batch
    int inputMemoryOffset = 0;
    int denseKernelMemoryOffset = 0;
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        // 1. Read in a single input value
        // 2. Multiply input value with entire 'row' of dense kernel values
        // 3. Add answer to output feature 
        for (int inputIndex = 0; inputIndex < denseLayerValues->inputNum; inputIndex++) {
            inputMemoryOffset = inputIndex + batchIndex*denseLayerValues->inputNum;
            for (int outputIndex = 0; outputIndex < denseLayerValues->outputNum; outputIndex++) {

                denseKernelMemoryOffset = outputIndex + inputIndex*denseLayerValues->outputNum;

                *(outputFeaturesReal + outputIndex + batchIndex*denseLayerValues->outputNum) += *(fMap->realValPtr + inputMemoryOffset) * *(denseLayerValues->kernelsReal + denseKernelMemoryOffset);
                *(outputFeaturesImag + outputIndex + batchIndex*denseLayerValues->outputNum) += *(fMap->imagValPtr + inputMemoryOffset) * *(denseLayerValues->kernelsImag + denseKernelMemoryOffset);

                // Add bias only once to each input
                if (inputIndex == 0) {
                    *(outputFeaturesReal + outputIndex + batchIndex*denseLayerValues->outputNum) += *(denseLayerValues->biasReal + outputIndex);
                    *(outputFeaturesImag + outputIndex + batchIndex*denseLayerValues->outputNum) += *(denseLayerValues->biasImag + outputIndex);
                }
            }

        } // Input Value
    } // Batch


    // Point F-Map to output values
    free(fMap->realValPtr);
    free(fMap->imagValPtr);
    fMap->realValPtr = outputFeaturesReal;
    fMap->imagValPtr = outputFeaturesImag;

    // Update size of F-Map
    fMap->H = 1;
    fMap->W = 1;
    fMap->C = denseLayerValues->outputNum;

    return;
}

/*
fMap: Input Feature Maps
probOut: Output Probabilities
*/
void softmax(FeatureMap *fMap, float *probOut) {


    probOut = (float*) malloc(fMap->B * FC_OUT_NEURONS * sizeof(float));

    float sum;
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        sum = 0.0;

        // Numerator
        for (int inputIndex = 0; inputIndex < FC_OUT_NEURONS; inputIndex++) {
            sum += exp(*(fMap->realValPtr + inputIndex + batchIndex*FC_OUT_NEURONS));
        }

        // Denominator
        for (int outputIndex = 0; outputIndex < FC_OUT_NEURONS; outputIndex++) {
            *(probOut + outputIndex + batchIndex*FC_OUT_NEURONS) = exp(*(fMap->realValPtr + outputIndex + batchIndex*FC_OUT_NEURONS)) / sum;
        }

    }

}

/*
fMap: Input Feature Maps
layerValues: Kernels and Bias values for output layer
probOut: Output Probabilities
*/
void outputLayer(FeatureMap *fMap, OutputLayerValues *layerValues, float **probOut) {

    // Allocate memory for output probabilities (also store intermediary values here); Init to zero
    *probOut = (float*) calloc(fMap->B * FC_OUT_NEURONS, sizeof(float));


    // Treat Feature Maps As Concatenated
    // ---- Dense Layer
    // float inputVal1, inputVal2, kernelVal1, kernelVal2, biasBal = 0;
    int inputMemoryOffset = 0;
    int denseKernelMemoryOffset1, denseKernelMemoryOffset2 = 0; // 2 will be half memory block from 1
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {

        
        for (int inputIndex = 0; inputIndex < layerValues->inputNum/2; inputIndex++) {
            inputMemoryOffset = inputIndex + batchIndex*(layerValues->inputNum/2);

            for (int outputIndex = 0; outputIndex < layerValues->outputNum; outputIndex++) {
                denseKernelMemoryOffset1 = outputIndex + inputIndex*(layerValues->outputNum);
                denseKernelMemoryOffset2 = denseKernelMemoryOffset1 + (layerValues->inputNum/2)*(layerValues->outputNum); // Add half block to memory offset 1


                // inputVal1 = *(fMap->realValPtr + inputMemoryOffset);
                // inputVal2 = *(fMap->imagValPtr + inputMemoryOffset);
                // kernelVal1 = *(layerValues->kernelsPtr + denseKernelMemoryOffset1);
                // kernelVal2 = *(layerValues->kernelsPtr + denseKernelMemoryOffset2);

                *(*probOut + outputIndex + batchIndex*layerValues->outputNum) += *(fMap->realValPtr + inputMemoryOffset) * *(layerValues->kernelsPtr + denseKernelMemoryOffset1);
                *(*probOut + outputIndex + batchIndex*layerValues->outputNum) += *(fMap->imagValPtr + inputMemoryOffset) * *(layerValues->kernelsPtr + denseKernelMemoryOffset2);

                // Add bias only once to each input
                if (inputIndex == 0) {
                    *(*probOut + outputIndex + batchIndex*layerValues->outputNum) += *(layerValues->biasPtr + outputIndex);
                }


            } // Output
        } // Input
    } // Batch


    // Treat Feature Maps As Concatenated
    // ---- Softmax
    float sum;
    float temp;
    for (int batchIndex = 0; batchIndex < fMap->B; batchIndex++) {
        sum = 0.0;

        // Numerator
        for (int inputIndex = 0; inputIndex < FC_OUT_NEURONS; inputIndex++) {
            sum += exp( *(*probOut + inputIndex + batchIndex*FC_OUT_NEURONS) );
        }

        // Denominator
        for (int outputIndex = 0; outputIndex < FC_OUT_NEURONS; outputIndex++) {
            temp = exp(*(*probOut + outputIndex + batchIndex*FC_OUT_NEURONS));
            *(*probOut + outputIndex + batchIndex*FC_OUT_NEURONS) = temp / sum;
        }

    } // Batch

}