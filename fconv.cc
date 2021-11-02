#include <iostream>
#include <assert.h>
#include <ctime>
#include <windows.h>
using namespace std;

void print(string s ,float* a, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            cout << a[i * n + j] << ' ';
        cout << endl;
    }
}

void im2col(float* in, float* out, int M, int U)
{
    // M*M U*U -> ((M-U+1)*(M-U+1)) * (U*U)
    for (int i = 0; i < (M-U+1)*(M-U+1); ++i)
        for (int j = 0; j < U*U; ++j)
        {
            int x_base = i / (M-U+1), y_base = i % (M-U+1);
            out[i * (U*U) + j] = in[(x_base + j / U) * M + (y_base + j % U)];
        }
}

inline void inner(float* K, float* W, float* O)
{
    float m0 = (K[0] - K[2]) * W[0];
    float m1 = (K[1] + K[2]) * (W[0] + W[1] + W[2]) / 2;
    float m2 = (K[2] - K[1]) * (W[0] - W[1] + W[2]) / 2;
    float m3 = (K[1] - K[5]) * W[2];

    O[0] = m0 + m1 + m2;
    O[1] = m1 - m2 - m3;
}

void conv(float* C, float* W, float* O)
{
    float* K0 = (float*)malloc(2*3*sizeof(float));
    float* K1 = (float*)malloc(2*3*sizeof(float));
    float* K2 = (float*)malloc(2*3*sizeof(float));
    float* K3 = (float*)malloc(2*3*sizeof(float));

    float* W0 = (float*)malloc(3*sizeof(float));
    float* W1 = (float*)malloc(3*sizeof(float));
    float* W2 = (float*)malloc(3*sizeof(float));
    float* W3 = (float*)malloc(3*sizeof(float));
    
    float* M0 = (float*)malloc(2*sizeof(float));
    float* M1 = (float*)malloc(2*sizeof(float));
    float* M2 = (float*)malloc(2*sizeof(float));
    float* M3 = (float*)malloc(2*sizeof(float));

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
        {
            int idxk = i * 3 + j;
            int idxc = i * 9 + j;
            K0[idxk] = C[idxc] - C[idxc + 6];
            K1[idxk] = C[idxc + 3] + C[idxc + 6];
            K2[idxk] = C[idxc + 6] - C[idxc + 3];
            K3[idxk] = C[idxc + 3] - C[idxc + 2 * 9 + 6];
        }
    for (int j = 0; j < 3; ++j)
    {
        W0[j] = W[j];
        W1[j] = (W[j] + W[j + 3] + W[j + 6]) / 2;
        W2[j] = (W[j] - W[j + 3] + W[j + 6]) / 2;
        W3[j] = W[j + 6];
    }

    inner(K0, W0, M0);
    inner(K1, W1, M1);
    inner(K2, W2, M2);
    inner(K3, W3, M3);

    for(int i = 0; i < 2; ++i)
    {
        O[i] = M0[i] + M1[i] + M2[i];
        O[i + 2] = M1[i] - M2[i] - M3[i];
    }
}

int main()
{
    srand(time(0));
    int count = 0;
    const int M = 4, U = 3;
    float* X = (float*)malloc(M*M*sizeof(float));
    float* W = (float*)malloc(U*U*sizeof(float));
    float* C = (float*)malloc((M-U+1)*(M-U+1)*U*U*sizeof(float));
    float* O = (float*)malloc((M-U+1)*(M-U+1)*sizeof(float));
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            X[i * M + j] = count++;
            cout.width(2);
            cout << X[i * M + j] << ' ';
        }
        cout << endl;
    }
    count = 0;
    for (int i = 0; i < U; ++i)
    {
        for (int j = 0; j < U; ++j)
        {
            W[i * U + j] = 1;
            cout.width(2);
            cout << W[i * U + j] << ' ';
        }
        cout << endl;
    }
    

    float OC[4] = {0};
    LARGE_INTEGER cpuFreq;
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    double runTime = 0.0;

    QueryPerformanceFrequency(&cpuFreq);

    QueryPerformanceCounter(&startTime);
    
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int u = 0; u < 3; ++u)
                for (int v = 0; v < 3; ++v)
                {
                    OC[i*2+j] += X[(i+u) * 4 + (j+v)] * W[u*3+v];
                }

    QueryPerformanceCounter(&endTime);
    runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
    cout << "ordinary: " << runTime << " ms" << endl;

    QueryPerformanceCounter(&startTime);

    im2col(X, C, M, U);
    conv(C, W, O);

    QueryPerformanceCounter(&endTime);
    runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
    cout << "fast: " << runTime << " ms" << endl;

    for (int i = 0; i < (M-U+1)*(M-U+1); ++i)
    {
        for (int j = 0; j < U*U; ++j)
        {
            cout.width(2);
            cout << C[i * U*U + j] << ' ';
        }
        cout << endl;
    }
    for (int i = 0; i < (M-U+1)*(M-U+1); ++i)
        cout << OC[i] << ' ';
    cout << endl;
    for (int i = 0; i < (M-U+1)*(M-U+1); ++i)
        cout << O[i] << ' ';
    return 0;
}
