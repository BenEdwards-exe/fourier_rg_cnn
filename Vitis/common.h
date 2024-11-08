#ifndef _COMMON_H_
#define _COMMON_H_

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"





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

#define CONV5_H_OUT 4
#define CONV5_W_OUT 4

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

///
extern "C" {

void elmnt_stream_krnl(
    // FMap Ptr In
    float* fMapInPtrReal,
    float* fMapInPtrImag,
    // FMap Ptr Out
    float* fMapOutPtrReal,
    float* fMapOutPtrImag,
    // Filter Ptrs
    float* filterPtrReal,
    float* filterPtrImag,
    // Sizes
    const uint32_t fMapSize,
    const uint32_t channelSizeIn,
    const uint32_t channelSizeOut
);

}


#endif