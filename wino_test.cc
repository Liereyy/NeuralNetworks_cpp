#include <iostream>
#include <assert.h>
#include <ctime>
#include <windows.h>
#include <cstring>
#include <cmath>
#include "winograd_conv.hpp"
using namespace std;

// e.g. const int K = 36, C = 16, H = 32, r = 3, m = 4;
const int K = 16, C = 8, H = 14, r = 3, m = 4;
const int tile_h = ceil((float)H / m), tile_w = tile_h;
const int P = tile_h * tile_w;
const int alpha = m + r - 1;

#define RAND_GEN (rand() % 10)

int main()
{
    LARGE_INTEGER cpuFreq;
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    LARGE_INTEGER startTime1;
    LARGE_INTEGER endTime1;
    double runTime = 0.0;
    double runTime1 = 0.0;

    float* d = (float*)malloc(C*(H+r-1)*(H+r-1)*sizeof(float));
    float* g = (float*)malloc(K*C*r*r*sizeof(float));
    float* h = (float*)malloc(K*H*H*sizeof(float));
    float* hs = (float*)malloc(K*H*H*sizeof(float));
    memset(hs, 0, K*H*H*sizeof(float));
    float* c = (float*)malloc(C*P*alpha*alpha*sizeof(float));

    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H+r-1; ++i)
            for (int j = 0; j < H+r-1; ++j)
                d[c * (H+r-1) * (H+r-1) + i * (H+r-1) + j] = RAND_GEN;

    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < r; ++j)
                    g[k * C * r * r + c * r * r + i * r + j] = RAND_GEN;

    QueryPerformanceFrequency(&cpuFreq);
    QueryPerformanceCounter(&startTime);

    ordinary_conv(d, g, hs, K, C, H, r, m);
    QueryPerformanceCounter(&endTime);
    runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
    cout << "normal: " << runTime << " ms" << endl;


    QueryPerformanceCounter(&startTime1);
    
    // print_tensor("d:", d, H+r-1, H+r-1);
    im2col(d, c, K, C, H, r, m);
    // print_tensor("c:\n", c, P, alpha*alpha);
    // cout << "tranform finished.\n";
    winograd_conv(c, g, h, K, C, H, r, m);
    // print_tensor("h:", h, H, H);

    QueryPerformanceCounter(&endTime1);
    runTime1 = (((endTime1.QuadPart - startTime1.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
    cout << "winograd: " << runTime1 << " ms" << endl;

    cout << runTime1 / runTime << endl;

    for (int i = 0; i < K*H*H; ++i)
        if (fabs(h[i] - hs[i]) > 1e-1)
        {
            cout << i << ": " << h[i] << ' ' << hs[i] << endl;
            cout << "test fail.\n";
            return 0;
        }
    cout << "test pass.\n";
}