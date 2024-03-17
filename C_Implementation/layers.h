#include <iostream>
#include <fstream>

/// ------------------------ Defines ------------------------ ///
// INPUTS:
#define INPUT_B 1
#define INPUT_H 150
#define INPUT_W 150
#define INPUT_C 1

// LAYERS:

// Layer1
#define CONV1_H 150
#define CONV1_W 150
#define CONV1_C_IN 1
#define CONV1_C_OUT 32
// Layer2
#define CONV2_H 75
#define CONV2_W 75
#define CONV2_C_IN 32
#define CONV2_C_OUT 32
// Layer3
#define CONV3_H 37
#define CONV3_W 37
#define CONV3_C_IN 32
#define CONV3_C_OUT 64
// Layer4
#define CONV4_H 18
#define CONV4_W 18
#define CONV4_C_IN 64
#define CONV4_C_OUT 64
// Layer5
#define CONV5_H 9
#define CONV5_W 9
#define CONV5_C_IN 64
#define CONV5_C_OUT 128

// FC1
#define FC1_INPUT_NUM 2048
#define FC1_NEURONS 512 
// FC2
#define FC2_NEURONS 256
// FC3
#define FC3_NEURONS 256
// OUTPUT
#define FC_OUT_INPUT_NUM 512
#define FC_OUT_NEURONS 3


/// ------------------------ Typedefs ------------------------ ///
typedef struct FeatureMap {
    int B,H,W,C = 0; // Shape: (Batch, Height, Width, Channel)
    float * realValPtr; // Pointer to Real F-Map
    float * imagValPtr; // Pointer to Imag F-Map
} FeatureMap;

typedef struct FiltersFreq {
    int H,W,cIn,cOut = 0; // Shape: (Height, Width, Channel_In, Channel_Out)
    float * realFilterPtr; // Pointer to Real Filters
    float * imagFilterPtr; // Pointer to Imag Filters
} FilterFreq;

typedef struct BatchNormLayerValues {
    float const epsilon = 0.001;
    int outputChannels = 0;
    // Following pointers so that the size of each layer's variables can be dynamically allocated
    float *gammaReal, *betaReal, *movingMeanReal, *movingVarReal;
    float *gammaImag, *betaImag, *movingMeanImag, *movingVarImag;
} BatchNormLayerValues;

typedef struct DenseLayerValues {
    int inputNum = 0; 
    int outputNum = 0; // amount of neurons in the layer
    float *kernelsReal, *kernelsImag;
    float *biasReal, *biasImag;
} DenseLayerValues;

typedef struct OutputLayerValues {
    int inputNum = 0;
    int outputNum = 0;
    float *kernelsPtr, *biasPtr;
} OutuptLayerValues;



/// ------------------------ Prototypes ------------------------ ///

// ------------------------- Functions that read in values ------------------------ //

/*
realFp: File Pointer to real input
imagFP: File Pointer to imag input
fmap: FeatureMap used for entire network
*/
void readInput(FILE **realFp, FILE **imagFp, FeatureMap *fMap);


/*
realFilterFp: File Pointer to real filter values for layer
imagFilterFp: File Pointer to imag filter values for layer
filters: FiltersFreq used for layer
*/
void readFilters(FILE **realFilterFp, FILE **imagFilterFp, FiltersFreq *filters, int layerNum);


/*
filePtrs: Array of pointers to FILE pointers
bNormLayerValues: Pointer to Batch Norm Layer Values struct
*/
void readBatchNormLayerValues(FILE *realFilePtrs[4], FILE *imagFilePtrs[4], BatchNormLayerValues *bNormLayerValues, int layerChannels);

/*
filePtrs: Array of pointers to FILE pointers
denseLayerValues: Pointer to Dense Layer Values struct
*/
void readDenseLayerValues(FILE *realFilePtrs[2], FILE *imagFilePtrs[2], DenseLayerValues *denseLayerValues, int denseLayerNum);

/*
filePtrs: Array of pointers to FILE pointers
outputLayerValues: Pointer to Output Layer Values struct
*/
void readOutputLayerValues(FILE *outFilePtrs[2], OutputLayerValues *outputLayerValues);

// ------------------------- Functions that perform layer calculations ------------ //

/*
fMap: Input feature maps
filters: Frequency domain filters for layer
Output is stored in fMap
*/
void elementWiseMultiply(FeatureMap *fMap, FilterFreq *filters);

/*
fMap: Input Feature Maps (BHWC)
bNormLayerValues: Pointer to Batch Norm Layer Values struct
*/
void batchNormLayer(FeatureMap *fMap, BatchNormLayerValues *bNormLayerValues);

/*
fMap: Input Feature Maps (BHWC)
Alpha value set as 0.2
*/
void leakyReLuLayer(FeatureMap *fMap);

/*
fMap: Input Feature Maps (BHWC)
Pool size: (2,2)
Stride: (2,2)
Padding: valid
*/
void maxPool2D(FeatureMap *fMap);

/*
fMap: Input Feature Maps (BHWC) [treated as "flattened"]
denseLayerValues: Kernels and Bias values for dense layer
*/
void denseLayer(FeatureMap *fMap, DenseLayerValues *denseLayerValues);

/*
fMap: Input Feature Maps
probOut: Output Probabilities
*/
void softmax(FeatureMap *fMap, float *probOut);

/*
fMap: Input Feature Maps
layerValues: Kernels and Bias values for output layer
probOut: Output Probabilities
*/
void outputLayer(FeatureMap *fMap, OutputLayerValues *layerValues, float **probOut);