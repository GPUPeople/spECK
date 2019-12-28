#pragma once
#include "stdint.h"

struct WorkDistributionConfig
{
    uint32_t threadsPerNnzOffset;
    uint32_t add3MinLength;
    uint32_t add2MinLength;
    uint32_t add1MinLength;
    uint32_t add3MaxCols;
    uint32_t add2MaxCols;
    uint32_t add1MaxCols;
    uint32_t sub2MaxCols;
    uint32_t sub1MaxCols;
    uint32_t sub2MinThreads;
    uint32_t sub1MinThreads;
    uint32_t add2MinConcurrentOps;
    uint32_t add1MinConcurrentOps;
    float maxOpsWeight64;
    float maxOpsWeight128;
    float maxOpsWeight256;
    float maxOpsWeight512;
    float maxOpsWeight1024;
    int staticThreadsPerRow;
};

const int layer1inputs = 6;
const int layer1outputs = 5;
const float layer1weights[30] = {
    -1.4186962,
    0.07587334,
    -1.7805182,
    0.04314838,
    -0.6445114,

    -0.13512687,
    0.04315747,
    -0.17808716,
    -0.04465475,
    -0.066692226,

    0.0752962,
    -0.104078434,
    0.16903225,
    -0.014818254,
    0.041726623,

    0.60707116,
    0.5149234,
    0.036716104,
    -0.070126966,
    0.37001306,

    -0.18412519,
    -0.11984752,
    -0.0021386633,
    -0.046877146,
    -0.16237561,

    0.8256881,
    0.7394887,
    0.07848209,
    0.058255166,
    0.9127046,

};
const float layer1offsets[5] = {
    20.326342,
    4.904586,
    15.644517,
    -0.018791957,
    17.353731,
};

const int layer2inputs = 5;
const int layer2outputs = 11;
const float layer2weights[55] = {
    1.1972289,
    1.9218329,
    3.7349546,
    9.81511,
    2.6227558,
    -6.426236,
    -43.629112,
    -22.914429,
    -0.781369,
    -0.45372608,
    -0.5380075,

    -5.0447526,
    -0.009247302,
    -0.008086299,
    -7.826033,
    -0.09450004,
    -0.11218474,
    -0.027270528,
    -0.0042230682,
    -0.0005600392,
    0.0005161164,
    0.004799865,

    3.116825,
    -0.26401007,
    -1.3744258,
    -8.859539,
    -140.25873,
    -18.334105,
    -8.593332,
    -1.0478442,
    -1.3645409,
    -1.2001375,
    -0.9904336,

    -0.0058234227,
    -0.063901104,
    -0.03595568,
    -0.008307365,
    -0.01685345,
    -0.012366829,
    0.028791403,
    0.024446918,
    0.028665425,
    -0.060744636,
    0.072986275,

    -0.069886416,
    -1.427334,
    -2.6894572,
    -1.6754347,
    0.0068549747,
    1.0374902,
    -0.71136826,
    -71.97164,
    -66.504616,
    -73.23368,
    -47.109512,

};
const float layer2offsets[11] = {
    -27.462452,
    -10.207701,
    -7.577945,
    -7.4156075,
    -3.9403565,
    -0.64222777,
    0.68561864,
    -1.2962743,
    -2.180761,
    -1.9912802,
    -4.157279,
};