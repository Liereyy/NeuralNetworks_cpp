#include <iostream>
#include <ctime>
#include <random>
#include <windows.h>
#include <string>
using namespace std;

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

int count = 0;

void print(string s ,float* a, int m, int n, int WA)
{
    cout << s << endl;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            cout << a[i * WA + j] << ' ';
        cout << endl;
    }
}

void CoppersmithWinograd_mm(float* A, float* B, float* C, int M, int N, int K, int WA, int WB)
{
    // cout << M << ' ' << N << ' ' << K << endl;
    if (M == 1 || N == 1 || K == 1 || M % 2 || N % 2 || K % 2)
    {
        // print("A:", A, M, N, WA);
        // print("B:", B, N, K, WB);
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

    free(S1);           S1=NULL;
    free(S2);           S2=NULL;
    free(S3);           S3=NULL;
    free(S4);           S4=NULL;
    free(T1);           T1=NULL;
    free(T2);           T2=NULL;
    free(T3);           T3=NULL;
    free(T4);           T4=NULL;
    free(M1);           M1=NULL;
    free(M2);           M2=NULL;
    free(M3);           M3=NULL;
    free(M4);           M4=NULL;
    free(M5);           M5=NULL;
    free(M6);           M6=NULL;
    free(M7);           M7=NULL;
}

#define RAND_GEN (rand() % 100 * 8)

int main()
{
    LARGE_INTEGER cpuFreq;
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    double runTime = 0.0;
    double runTime1 = 0.0;
    unsigned seed = time(0);
    srand(seed);
    while (1)
    {
        // const int M = RAND_GEN, N = RAND_GEN, K = RAND_GEN, rangeTop = 30;
        const int M = 144, N = 1600, K = 14400, rangeTop = 30;
        // cout << M << ' ' << N << ' ' << K << endl;
        float * mA = (float*) malloc(M*N*sizeof(float));
        float * mB = (float*) malloc(N*K*sizeof(float));
        float * mC = (float*) malloc(M*K*sizeof(float));
        float * mCS = (float*) malloc(M*K*sizeof(float));
        for(int j = 0; j < M*N; j++)
            mA[j] = rand() % rangeTop;
        // print("mA:", mA, M, N, N);
        for(int j = 0; j < N*K; j++)
            mB[j] = rand() % rangeTop;
        // print("mB:", mB, N, K, K);
        // for (int i = 0; i < M*N; ++i)
        //     cout << mA[i] << ' ';
        // cout << endl;
        // for (int i = 0; i < N*K; ++i)
        //     cout << mB[i] << ' ';
        // cout << endl;
        QueryPerformanceFrequency(&cpuFreq);
        QueryPerformanceCounter(&startTime);
        
        basic_mm(mA, mB, mCS, M, N, K, N, K);
        // for (int i = 0; i < M*K; ++i)
        //     cout << mCS[i] << ' ';
        // cout << endl;

        QueryPerformanceCounter(&endTime);
        runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
        cout<< "traditional: "  << runTime << " ms" << endl;

        QueryPerformanceCounter(&startTime);
        
        CoppersmithWinograd_mm(mA, mB, mC, M, N, K, N, K);

        // for (int i = 0; i < M*K; ++i)
        //     cout << mC[i] << ' ';
        // cout << endl;
        QueryPerformanceCounter(&endTime);
        runTime1 = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
        cout << "winograd: " << runTime1 << " ms" << endl;

        for (int i = 0; i < M*K; ++i)
            if (mC[i] != mCS[i])
            {
                cout << i;
                return 0;
            }
        cout << runTime1 / runTime << endl;
        cout << endl;
        Sleep(1500);
    }
}