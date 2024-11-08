#include "common.h"
#include "hls_math.h"
#include "hls_stream.h"
#include <cmath>
#include <cstdint>
#include <sys/types.h>



// For loop tripcount testing
#define FMAP_IN_SIZE_MAX 75
#define FMAP_IN_SIZE_MIN 75
#define CHANNEL_IN_SIZE 32
#define CHANNEL_OUT_SIZE 32



static void streamInFmaps(
    hls::stream<float>& inStream,
    float* dataInPtr,
    uint32_t fMapSize,
    uint32_t channelInSize
) {
#pragma HLS INLINE off
    uint32_t ptrIdx = 0;
    for (uint32_t h = 0; h < fMapSize; h++) {
    #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX
        for (uint32_t w = 0; w < fMapSize; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX
            for (uint32_t i = 0; i < channelInSize; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_IN_SIZE max=CHANNEL_IN_SIZE
            #pragma HLS PIPELINE II=1
                float valIn = dataInPtr[ptrIdx];
                ptrIdx += 1;
                inStream.write(valIn);
            }
        }
    }

}

static void streamInFilters(
    hls::stream<float>& inStream,
    float* dataInPtr,
    uint32_t fMapSize,
    uint32_t channelInSize,
    uint32_t channelOutSize
) {
#pragma HLS INLINE off
    uint32_t ptrIdx = 0;

    for (uint32_t h = 0; h < fMapSize; h++) {
    #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX
        for (uint32_t w = 0; w < fMapSize; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX

            for (uint32_t i = 0; i < channelInSize; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_IN_SIZE max=CHANNEL_IN_SIZE
                for (uint32_t j = 0; j < channelOutSize; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_OUT_SIZE max=CHANNEL_OUT_SIZE
                #pragma HLS PIPELINE II=1
                    float valIn = dataInPtr[ptrIdx];
                    ptrIdx += 1;
                    inStream.write(valIn);
                }
            }

        }
    }


}


static void streamOutFmaps(
    hls::stream<float>& outStream,
    float* dataOutPtr,
    uint32_t fMapSize,
    uint32_t channelSizeOut
) {
#pragma HLS INLINE off
    uint32_t ptrIdx = 0;

    for (uint32_t h = 0; h < fMapSize; h++) {
    #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX
        for (uint32_t w = 0; w < fMapSize; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX

            for (uint32_t i = 0; i < channelSizeOut; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_OUT_SIZE max=CHANNEL_OUT_SIZE
            #pragma HLS PIPELINE II=1
                float valOut = outStream.read();
                dataOutPtr[ptrIdx] = valOut;
                ptrIdx += 1;
            }

        }
    }


}




static void complexMulAcc(
    hls::stream<float>& fMapInReal,
    hls::stream<float>& fMapInImag,
    hls::stream<float>& filterInReal,
    hls::stream<float>& filterInImag,
    hls::stream<float>& fMapOutReal,
    hls::stream<float>& fMapOutImag,
    const uint32_t fMapSize,
    const uint32_t channelInSize,
    const uint32_t channelOutSize
) {
#pragma HLS INLINE off
    // Maximum input or output channels supported: 128
    float bufferOutReal[128];
    float bufferOutImag[128];

    for (uint32_t h = 0; h < fMapSize; h++) {
    #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX
        for (uint32_t w = 0; w < fMapSize; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=FMAP_IN_SIZE_MAX max=FMAP_IN_SIZE_MAX

            for (uint32_t c_in = 0; c_in < channelInSize; c_in++) {
            #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_IN_SIZE max=CHANNEL_IN_SIZE

                float fMapValReal = fMapInReal.read();
                float fMapValImag = fMapInImag.read();

                for (uint32_t c_out = 0; c_out < channelOutSize; c_out++) {
                #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_OUT_SIZE max=CHANNEL_OUT_SIZE
                #pragma HLS PIPELINE II=1

                    float filterValReal = filterInReal.read();
                    float filterValImag = filterInImag.read();

                    //  Complex Multiply
                    float AC, BD, AD, BC, ansReal, ansImag;
                    AC = fMapValReal * filterValReal;
                    BD = fMapValImag * filterValImag;
                    AD = fMapValReal * filterValImag;
                    BC = fMapValImag * filterValReal;

                    ansReal = AC + BD;
                    ansImag = BC - AD;

                    // Accumulate
                    float prevReal = (c_in == 0) ? 0 : bufferOutReal[c_out];
                    float prevImag = (c_in == 0) ? 0 : bufferOutImag[c_out];
                    float accReal = prevReal + ansReal;
                    float accImag = prevImag + ansImag;

                    // Write to buffer
                    bufferOutReal[c_out] = accReal;
                    bufferOutImag[c_out] = accImag;

                } // Channel Out
            } // Channel In

write_out_streams:
            for (uint32_t writeOutIdx = 0; writeOutIdx < channelOutSize; writeOutIdx++) {
            #pragma HLS LOOP_TRIPCOUNT min=CHANNEL_OUT_SIZE max=CHANNEL_OUT_SIZE
            #pragma HLS PIPELINE II=1
                float outReal = bufferOutReal[writeOutIdx];
                float outImag = bufferOutImag[writeOutIdx];
                fMapOutReal.write(outReal);
                fMapOutImag.write(outImag);
            } // write out idx

        } // Width

    } // Height



}



/// -- KERNEL TOP FUNCTION --- ///
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
) {
#pragma HLS INTERFACE m_axi port=fMapInPtrReal bundle=gmem0
#pragma HLS INTERFACE m_axi port=fMapInPtrImag bundle=gmem1

#pragma HLS INTERFACE m_axi port=fMapOutPtrReal bundle=gmem0
#pragma HLS INTERFACE m_axi port=fMapOutPtrImag bundle=gmem1

#pragma HLS INTERFACE m_axi port=filterPtrReal bundle=gmem4
#pragma HLS INTERFACE m_axi port=filterPtrImag bundle=gmem5

#pragma HLS INTERFACE s_axilite port=fMapSize 
#pragma HLS INTERFACE s_axilite port=channelSizeIn
#pragma HLS INTERFACE s_axilite port=channelSizeOut

#pragma HLS INTERFACE s_axilite port=return


#pragma HLS STABLE variable = fMapInPtrReal
#pragma HLS STABLE variable = fMapInPtrImag
#pragma HLS STABLE variable = fMapOutPtrReal
#pragma HLS STABLE variable = fMapOutPtrImag
#pragma HLS STABLE variable = filterPtrReal
#pragma HLS STABLE variable = filterPtrImag


    hls::stream<float> fMapStreamInReal, fMapStreamInImag;
    hls::stream<float> filtersInReal, filtersInImag;
    hls::stream<float> fMapStreamOutReal, fMapStreamOutImag;
    

#pragma HLS DATAFLOW // Will run functions in parallel

    streamInFmaps(fMapStreamInReal, fMapInPtrReal, fMapSize, channelSizeIn);
    streamInFmaps(fMapStreamInImag, fMapInPtrImag, fMapSize, channelSizeIn);
    
    streamInFilters(filtersInReal, filterPtrReal, fMapSize, channelSizeIn, channelSizeOut);
    streamInFilters(filtersInImag, filterPtrImag, fMapSize, channelSizeIn, channelSizeOut);

    complexMulAcc(fMapStreamInReal, fMapStreamInImag, filtersInReal, filtersInImag, fMapStreamOutReal, fMapStreamOutImag, fMapSize, channelSizeIn, channelSizeOut);

    streamOutFmaps(fMapStreamOutReal, fMapOutPtrReal, fMapSize, channelSizeOut);
    streamOutFmaps(fMapStreamOutImag, fMapOutPtrImag, fMapSize, channelSizeOut);


}


}
