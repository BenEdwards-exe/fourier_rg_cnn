#include <iostream>
#include <fstream>
#include <cmath>




typedef struct BatchNormLayerValues
{ // Variables used for batch normalization layer
    float const epsilon = 0.001;
    float outputChannels;
    // All of these are pointers so that the size of each layers variables can
    // be dynamically allocated.
    float *gamma;
    float *beta;
    float *movingMean;
    float *movingVar;

} BatchNormLayerValues;


// featureMap shape: H,W,C
// bnorm variables shape: C
void batchNormalize(float *featureMap, BatchNormLayerValues *layerVariables) {

    // Formula: gamma * ((featureMap - moving_mean) / sqrt(moving_var + 0.001)) + beta

    // for each H,W channel, apply bnorm
    int filtersH = 150; // height
    int filtersW = 150; // width
    int cOut = layerVariables->outputChannels;

    float gamma, fMapVal, movingMean, movingVar, beta = 0;


    for (int i = 0; i < filtersH; i++) {
        for (int j = 0; j < filtersW; j++) {
            for (int k = 0; k < layerVariables->outputChannels; k++) {
                gamma = *(layerVariables->gamma + k);
                movingMean = *(layerVariables->movingMean + k);
                movingVar = *(layerVariables->movingVar + k);
                beta = *(layerVariables->beta + k);
                fMapVal = *(featureMap + i*filtersH*cOut + j*cOut + k);
                // TODO: have the sqrt pre-calculated
                *(featureMap + i*filtersH*cOut + j*cOut + k) = gamma * ((fMapVal - movingMean) / sqrt(movingVar + 0.001)) + beta;
            }
        }
    }

}

// Return pointer to new feature map
float* maxPooling2D(float *featureMap) {
    // Pool size = (2,2); Stride = 2 (keras: if stride is None, default to pool size)
    // Padding = Valid

    int filtersH = 150;
    int filtersW = 150;
    int cOut = 32; // Same as cIn

    int outH = floor((filtersH - 2) / 2) + 1;
    int outW = floor((filtersW - 2) / 2) + 1;

    float *featureMapOut = (float *) malloc(outH*outW*cOut * sizeof(float));


    // Temp values, TODO: Remove and integrate
    float valIn0, valIn1, valIn2, valIn3 = 0;
    int x, y = 0;
    float valOut = 0;

    // Loop through output map
    // Out[a][b] =  max( max(In[2a][2b],In[2a,2b+1]), max(In[2a+1][2b],In[2a+1][2b+1]) ) 
    for (int i = 0; i < outH; i++) {
        for (int j = 0; j < outW; j++) {
            for (int k = 0; k < cOut; k++) {

                x = i*2; 
                y = j*2;
                valIn0 = *(featureMap + x*filtersH*cOut + y*cOut + k);

                x = i*2;
                y = j*2 + 1;
                valIn1 = *(featureMap + x*filtersH*cOut + y*cOut + k);

                x = i*2 + 1;
                y = j*2;
                valIn2 = *(featureMap + x*filtersH*cOut + y*cOut + k);

                x = i*2 + 1;
                y = j*2 + 1;
                valIn3 = *(featureMap + x*filtersH*cOut + y*cOut + k);

                valOut = fmax(fmax(valIn0, valIn1),fmax(valIn2, valIn3));
                *(featureMapOut + i*outH*cOut + j*cOut + k) = valOut;
            }
        }
    }


    // TODO: Free pointer memory



    // Point feature map to the output
    return featureMapOut;
}


void leakyReLu(float *featureMap, float alpha) {

    // TODO: these should be dynamic
    int filtersH = 150;
    int filtersW = 150;
    int cOut = 32;

    float fMapVal = 0;

    for (int i = 0; i < filtersH; i++) {
        for (int j = 0; j < filtersW; j++) {
            for (int k = 0; k < cOut; k++) {
                fMapVal = *(featureMap + i*filtersH*cOut + j*cOut + k);
                *(featureMap + i*filtersH*cOut + j*cOut + k) = fmax(alpha*fMapVal, fMapVal);
            }
        }
    }
    

}

void readBatchNormalizationWeights(BatchNormLayerValues *layerVariables) {
    layerVariables->outputChannels = 32;

    // Allocate memory
    layerVariables->gamma = (float *) malloc(32 * sizeof(float));
    layerVariables->beta = (float *) malloc(32 * sizeof(float));
    layerVariables->movingMean = (float *) malloc(32 * sizeof(float));
    layerVariables->movingVar = (float *) malloc(32 * sizeof(float));

    // File Pointers
    // TODO: Make these pointers arguments to function
    FILE *gammaPtr = fopen("b_norm_real_gamma.txt", "r");
    FILE *betaPtr = fopen("b_norm_real_beta.txt", "r");
    FILE *movintMeanPtr = fopen("b_norm_real_moving_mean.txt", "r");
    FILE *movingVarPtr = fopen("b_norm_real_moving_var.txt", "r");

    // Read weights from text files
    for (int i = 0; i < 32; i++) {
        fscanf(gammaPtr, "%f", (layerVariables->gamma + i));
        fscanf(betaPtr, "%f", (layerVariables->beta + i));
        fscanf(movintMeanPtr, "%f", (layerVariables->movingMean + i));
        fscanf(movingVarPtr, "%f", (layerVariables->movingVar + i));
    }

}



// Read a single image from the text file
// Updates the file pointer
void readImage(FILE **fp, float imageArr[150][150]) {
    float temp = 0;
    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 150; j++) {
            fscanf(*fp, "%f", &temp);
            imageArr[i][j] = temp;
        }
    }
    return;
}

// Will read the next filter with every call
// TODO: find a way to read specific filter
void readFilter(FILE **fpReal, FILE**fpImag, float realArr[150][150], float imagArr[150][150], int *currentFilter) {
    *currentFilter += 1;
    float temp = 0;
    for (int i = 0; i < 150; i++) {
        for (int j = 0; j < 150; j++) {
            fscanf(*fpReal, "%f", &temp);
            realArr[i][j] = temp;
            fscanf(*fpImag, "%f", &temp);
            imagArr[i][j] = temp;
        }
    }
    return;
}



int fourierConvLayer() {

    // File pointers
    FILE *realFp = fopen("x_test_real.txt", "r"); // real images
    FILE *imagFp = fopen("x_test_imag.txt", "r"); // imag images
    FILE *filtersRealFp = fopen("weights_real.txt", "r"); // real filters
    FILE *filtersImagFp = fopen("weights_imag.txt", "r"); // imag filters

    // Read a single image
    float imageReal[150][150] = {0};
    float imageImag[150][150] = {0};
    readImage(&realFp, imageReal);
    readImage(&imagFp, imageImag);

    // Pointers to memory where all filters are stored
    float * filtersRealPtr = (float*) malloc(150*150*32 * sizeof(float));
    float * filtersImagPtr = (float*) malloc(150*150*32 * sizeof(float));

    // Pointers to memory where intermediary feature map results are stored
    float * featureMapReal = (float*) malloc(150*150*32 * sizeof(float));
    float * featureMapImag = (float*) malloc(150*150*32 * sizeof(float));

    int filtersH = 150; // height
    int filtersW = 150; // width
    int cOut = 32; // depth

    float temp = 0;
    // Read all real filters
    // Shape: H,W,C_out
    for (int i = 0; i < filtersH; i++) {
        for (int j = 0; j < filtersW; j++) {
            for (int k = 0; k < cOut; k++) {
                // x * Height * Depth + y * Depth + z;
                fscanf(filtersRealFp, "%f", &temp);
                *(filtersRealPtr + i*filtersH*cOut + j*cOut + k) = temp;
                fscanf(filtersImagFp, "%f", &temp);
                *(filtersImagPtr + i*filtersH*cOut + j*cOut + k) = temp;
            }
        }
    }



    // AC = Input_Real * Filter_Real
    // BD = Input_Imag * Filter_Imag
    // AD = Input_Real * Filter_Imag
    // BC = Input_Imag * Filter_Real

    float ac, bd, ad, bc = 0;
    float imageRealCurrent, imageImagCurrent, filterRealCurrent, filterImagCurrent = 0;
    float outRealCurrent, outImagCurrent = 0;
    for (int i = 0; i < filtersH; i++) {
        for (int j = 0; j < filtersW; j++) {
            imageRealCurrent = imageReal[i][j];
            imageImagCurrent = imageImag[i][j];


            for (int k = 0; k < cOut; k++) {
                filterRealCurrent = *(filtersRealPtr + i*filtersH*cOut + j*cOut + k);
                filterImagCurrent = *(filtersImagPtr + i*filtersH*cOut + j*cOut + k);

                ac = imageRealCurrent * filterRealCurrent;
                bd = imageImagCurrent * filterImagCurrent;
                ad = imageRealCurrent * filterImagCurrent;
                bc = imageImagCurrent * filterRealCurrent;

                outRealCurrent = ac + bd;
                outImagCurrent = bc - ad;

                // Store in Feature Map Buffer
                *(featureMapReal + i*filtersH*cOut + j*cOut + k) = outRealCurrent;
                *(featureMapImag + i*filtersH*cOut + j*cOut + k) = outImagCurrent;

                // TODO: reduce_sum section

                // TODO: append to list (if using batch)

            }
        }
    }

    BatchNormLayerValues bNormL0;
    readBatchNormalizationWeights(&bNormL0);

    batchNormalize(featureMapReal, &bNormL0);

    
    leakyReLu(featureMapReal, 0.2);



    // Testing stuff
    for (int i = 0; i < 32; i++) {
        std::cout << *(bNormL0.gamma + i) << std::endl;
    }


    featureMapReal = maxPooling2D(featureMapReal);

    // Testing stuff
    float currentFilter[32] = {0};
    filtersH = 75;
    filtersW = 75;
    for (int i = 0; i < filtersH; i++) {
        for (int j = 0; j < filtersW; j++) {
            for (int k = 0; k < cOut; k++) {
                if (i == 10 && j == 3) {
                    currentFilter[k] = *(featureMapReal + i*filtersH*cOut + j*cOut + k);
                }
                // currentFilter[k] = *(featureMapReal + i*filtersH*cOut + j*cOut + k);
            }
        }
    }





    return 1;

}

// int main() {
//     fourierConvLayer();

//     return;
// }