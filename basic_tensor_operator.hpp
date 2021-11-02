#pragma once
#include "basic_tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_tensor_gen.hpp"
#include <windows.h>
using namespace std;

LARGE_INTEGER cpuFreq_bto;
LARGE_INTEGER startTime_bto;
LARGE_INTEGER endTime_bto;
double runTime_bto;

// #define TIME_MM

// Coppersmith-Winograd
/*
* m*k k*n = m*k
* S1 = A21 + A22     T1 = B12 - B11
* S2 = S1 - A11      T2 = B22 - T1
* S3 = A11 - A21     T3 = B22 - B12
* S4 = A12 - S2      T4 = T2 - B21
* M1 = A11 * B11     U1 = M1 + M2
* M2 = A12 * B21     U2 = M1 + M6
* M3 = S4 * B22      U3 = U2 + M7
* M4 = A22 * T4      U4 = U2 + M5
* M5 = S1 * T1       U5 = U4 + M3
* M6 = S2 * T2       U6 = U3 - U4
* M7 = S3 * T3       U7 = U3 + M5
* C11 = U1
* C12 = U5
* C21 = U6
* C22 = U7
*/
void CoppersmithWinograd_mm(float* A, float* B, float* C, int M, int N, int K, int WA, int WB);
void basic_mm(float* A, float* B, float* C, int M, int N, int K, int WA, int WB);

namespace torch
{
    template <typename T>
    basic_tensor<T> operator*(float t1, basic_tensor<T> t2)
    {
        for (size_t i = 0; i < t2.data.size(); ++i)
            t2[i] *= t1;
        return t2;
    }

    template <typename T>
    basic_tensor<T> operator/(basic_tensor<T> t1, float t2)
    {
        for (size_t i = 0; i < t1.data.size(); ++i)
            t1[i] /= t2;
        return t1;
    }

    template <typename T>
    basic_tensor<T> operator+(basic_tensor<T> t1, float t2)
    {
        for (size_t i = 0; i < t1.data.size(); ++i)
            t1[i] += t2;
        return t1;
    }

    template <typename T>
    basic_tensor<T> operator+(basic_tensor<T> t1, basic_tensor<T> t2)
    {
        for (int i = 0; i < t1.data_size(); ++i)
            t1[i] += t2[i];
        return t1;
    }

    template <typename T>
    basic_tensor<T> operator-(basic_tensor<T> t1, basic_tensor<T> t2)
    {
        for (size_t i = 0; i < t1.data.size(); ++i)
            t1.data[i] -= t2.data[i];
        return t1;
    }

    // Coppersmith-Winograd
    // 仅处理二维矩阵乘法
    template <typename T>
    basic_tensor<T> operator*(basic_tensor<T> t1, basic_tensor<T> t2)
    {
        #ifdef TIME_MM
        QueryPerformanceFrequency(&cpuFreq_bto);
        QueryPerformanceCounter(&startTime_bto);
        #endif

        vector<int> dsize1(t1.dsize), dsize2(t2.dsize);
        if (dsize1[1] != dsize2[0])
        {
            cerr << "mm size mismatched.\n";
            exit(0);
        }
        int m = dsize1[0], n = dsize1[1], k = dsize2[1];
        basic_tensor<T> res = bzeros({m, k});
        // CoppersmithWinograd_mm(&t1.data[0], &t2.data[0], &res.data[0], m, n, k, n, k);
        basic_mm(&t1.data[0], &t2.data[0], &res.data[0], m, n, k, n, k);
        #ifdef TIME_MM
        QueryPerformanceCounter(&endTime_bto);
        runTime_bto = (((endTime_bto.QuadPart - startTime_bto.QuadPart) * 1000.0f) / cpuFreq_bto.QuadPart);
        cout << "mm: " << t1.dsize << ' ' << t2.dsize << "   compute consume = " << runTime_bto << " ms" << endl;
        #endif
        return res;
    }

    template <typename T>
    pair<size_t, size_t> max_index(basic_tensor<T> t)
    {
        size_t index = 0;
        for (size_t i = 1; i < t.data_size(); ++i)
            if (t[i] > t[index])
                index = i;
        return make_pair<size_t, size_t>(index / t.dsize[1], index % t.dsize[1]);
    }

    template <typename T>
    basic_tensor<T> soft_max(basic_tensor<T> t)
    {
        double sum = 0;
        for (int i = 0; i < t.data_size(); ++i)
        {
            t[i] = pow(consts::e, t[i] > 18 ? 18 : t[i]);
            sum += t[i];
        }
        for (int i = 0; i < t.data_size(); ++i)
            t[i] = t[i] / sum;
        return t;
    }

    template <typename T>
    basic_tensor<T> log(basic_tensor<T> t)
    {
        for (int i = 0; i < t.data_size(); ++i)
            t[i] = std::log(t[i]);
        return t;
    }

    template <typename T>
    double sum(basic_tensor<T>& t)
    {
        double res = 0;
        for (int i = 0; i < t.data_size(); ++i)
            res += t[i];
        return res;
    }
}

// C = A * B
void basic_mm(float* A, float* B, float* C, int M, int N, int K, int WA, int WB)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
        {
            C[i * K + j] = 0;
            for (int p = 0; p < N; ++p)
                C[i * K + j] += A[i * WA + p] * B[p * WB + j];
        }
}

void CoppersmithWinograd_mm(float* A, float* B, float* C, int M, int N, int K, int WA, int WB)
{
    if (M == 1 || N == 1 || K == 1 || M % 2 || N % 2 || K % 2)
    {
        basic_mm(A, B, C, M, N, K, WA, WB);
        return;
    }
    
    float* S1 = (float*)malloc((M/2) * (N/2) * sizeof(float));
    float* S2 = (float*)malloc((M/2) * (N/2) * sizeof(float));
    float* S3 = (float*)malloc((M/2) * (N/2) * sizeof(float));
    float* S4 = (float*)malloc((M/2) * (N/2) * sizeof(float));

    for(int i = 0; i < M/2; i++)
        for(int j = 0; j < N/2; j++)
        {
            int idxA, idxS = i * (N/2) + j;

            //S1     = A21 + A22
            idxA     = (i + (M/2)) * WA + j;
            S1[idxS] = A[idxA] + A[idxA + N/2];

            //S2     = S1 - A11
            idxA     = i * WA + j;
            S2[idxS] = S1[idxS] - A[idxA];

            //S3     = A11 - A21
            S3[idxS] = A[idxA] - A[idxA + (M/2) * WA];

            //S4     = A12 - S2
            idxA     = i * WA + (N/2) + j;
            S4[idxS] = A[idxA] - S2[idxS];
        }
    
    float* T1 = (float*) malloc((N/2) * (K/2) * sizeof(float));
    float* T2 = (float*) malloc((N/2) * (K/2) * sizeof(float));
    float* T3 = (float*) malloc((N/2) * (K/2) * sizeof(float));
    float* T4 = (float*) malloc((N/2) * (K/2) * sizeof(float));

    for(int i = 0; i < N/2; i++)
        for(int j = 0; j < K/2; j++)
        {
            int idxB, idxT = i * (K/2) + j;

            //T1     = B12 - B11
            idxB     = i * WB + j;
            T1[idxT] = B[idxB + (K/2)] - B[idxB];

            //T2     = B22 - T1
            idxB     = (i + (N/2)) * WB + (K/2) + j;
            T2[idxT] = B[idxB] - T1[idxT];

            //T3     = B22 - B12
            idxB     = i * WB + (K/2) + j;
            T3[idxT] = B[idxB + (N/2) * WB] - B[idxB];

            //T4     = T2 - B21
            idxB     = (i + (N/2)) * WB + j;
            T4[idxT] = T2[idxT] - B[idxB];
        }
    

    //M1 = A11 * B11
    float* M1 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(A, B, M1, M/2, N/2, K/2, WA, WB);

    //M2 = A12 * B21
    float* M2 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(A + N/2, B + (N/2) * WB, M2, M/2, N/2, K/2, WA, WB);

    //M3 = S4 * B22
    float* M3 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(S4, B + (N/2) * WB + (K/2), M3, M/2, N/2, K/2, N/2, WB);

    //M4 = A22 * T4
    float* M4 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(A + (M/2) * WA + (N/2), T4, M4, M/2, N/2, K/2, WA, K/2);

    //M5 = S1 * T1
    float* M5 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(S1, T1, M5, M/2, N/2, K/2, N/2, K/2);

    //M6 = S2 * T2
    float* M6 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(S2, T2, M6, M/2, N/2, K/2, N/2, K/2);

    //M7 = S3 * T3
    float* M7 = (float*)malloc((M/2) * (K/2) * sizeof(float));
    CoppersmithWinograd_mm(S3, T3, M7, M/2, N/2, K/2, N/2, K/2);


    //C11 = U1 = M1 + M2
    //C12 = U5 = U4 + M3 = U2 + M5 + M3 = M1 + M6 + M5 + M3
    //C21 = U6 = U3 - M4 = U2 + M7 - M4 = M1 + M6 + M7 - M4
    //C22 = U7 = U3 + M5 = U2 + M7 + M5 = M1 + M6 + M7 + M5
    for(int i = 0; i < M/2; i++)
        for(int j = 0; j < K/2; j++)
        {
            int idx = i * (K/2) + j;
            C[i * K + j] = M1[idx] + M2[idx];
            C[i * K + j + (K/2)] = M1[idx] + M6[idx] + M5[idx] + M3[idx];
            C[(i + (M/2)) * K + j] = M1[idx] + M6[idx] + M7[idx] - M4[idx];
            C[(i + (M/2)) * K + j + (K/2)] = M1[idx] + M6[idx] + M7[idx] + M5[idx];
        }

    free(S1);
    free(S2);
    free(S3);
    free(S4);
    free(T1);
    free(T2);
    free(T3);
    free(T4);
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
}