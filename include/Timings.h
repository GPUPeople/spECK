#pragma once
#include <vector>

struct Timings {
    bool measureAll = false;
    bool measureCompleteTime = false;
    float init = 0.0f;
    float countProducts = 0.0f;
    float loadBalanceCounting = 0.0f;
    float globalMapsCounting = 0.0f;
    float spGEMMCounting = 0.0f;
    float allocC = 0.0f;
    float loadBalanceNumeric = 0.0f;
    float globalMapsNumeric = 0.0f;
    float spGEMMNumeric = 0.0f;
    float sorting = 0.0f;
    float cleanup = 0.0f;
    float complete = 0.0f;

    void operator+=(const Timings& b) {
        init += b.init;
        countProducts += b.countProducts;
        loadBalanceCounting += b.loadBalanceCounting;
        globalMapsCounting += b.globalMapsCounting;
        spGEMMCounting += b.spGEMMCounting;
        allocC += b.allocC;
        loadBalanceNumeric += b.loadBalanceNumeric;
        globalMapsNumeric += b.globalMapsNumeric;
        spGEMMNumeric += b.spGEMMNumeric;
        sorting += b.sorting;
        cleanup += b.cleanup;
        complete += b.complete;
    }

    void operator/=(const float& x) {
        init /= x;
        countProducts /= x;
        loadBalanceCounting /= x;
        globalMapsCounting /= x;
        spGEMMCounting /= x;
        allocC /= x;
        loadBalanceNumeric /= x;
        globalMapsNumeric /= x;
        spGEMMNumeric /= x;
        sorting /= x;
        cleanup /= x;
        complete /= x;
    }
};

// Timings averageTimings(std::vector<Timings> timings) {
//     Timings combined;
//     for (auto t : timings) {
//         combined += t;
//     }
//     combined /= float(timings.size());

//     return combined;
// }