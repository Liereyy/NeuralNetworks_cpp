#include <iostream>
#include <assert.h>
#include <ctime>
#include <windows.h>
#include "basic_tensor_operator.hpp"
// #include "fast_mm.hpp"
#include <cstring>
#include <cmath>
using namespace std;

void print_tensor(string s, float* a, int m, int n)
{
    cout << s << endl;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            cout << a[i * n + j] << ' ';
        cout << '\\' << endl;
    }
}

void print_tensor(string s, float* a, int C, int m, int n)
{
    cout << s << endl;
    for (int c = 0; c < C; ++c)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
                cout << a[c * m * n + i * n + j] << ' ';
            cout << endl;
        }
        cout << endl;
    }
}


// 采用F(4x4, 3x3)

// alpha x r
float G[] =
{
    0.25,       0,          0,
    -1.0 / 6,   -1.0 / 6,   -1.0 / 6,
    -1.0 / 6,   1.0 / 6,   -1.0 / 6,
    1.0 / 24,   1.0 / 12,   1.0 / 6,
    1.0 / 24,   -1.0 / 12,  1.0 / 6,
    0,          0,          1
};

float GT[] =
{
    0.25, -1.0/6, -1.0/6, 1.0/24, 1.0/24, 0,
    0, -1.0/6, 1.0/6, 1.0/12, -1.0/12, 0,
    0, -1.0/6, -1.0/6, 1.0/6, 1.0/6, 1
};

// alpha x alpha
float B[] =
{
    4, 0, 0, 0, 0, 0,
    0, -4, 4, -2, 2, 4,
    -5, -4, -4, -1, -1, 0,
    0, 1, -1, 2, -2, -5,
    1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 1
};

float BT[] =
{
    4, 0, -5, 0, 1, 0,
    0, -4, -4, 1, 1, 0,
    0, 4, -4, -1, 1, 0,
    0, -2, -1, 2, 1, 0,
    0, 2, -1, -2, 1, 0,
    0, 4, 0, -5, 0, 1
};


// m x alpha
float A[] = 
{
    1, 0, 0, 0,
    1, 1, 1, 1,
    1, -1, 1, -1,
    1, 2, 4, 8,
    1, -2, 4, -8,
    0, 0, 0, 1
};

float AT[] = 
{
    1, 1, 1, 1, 1, 0,
    0, 1, -1, 2, -2, 0,
    0, 1, 1, 4, 4, 0,
    0, 1, -1, 8, -8, 1
};

void ordinary_conv(float* d, float* g, float* hs, int K, int C, int H, int r, int m)
{
    for (int oc = 0; oc < K; ++oc)
        for (int ic = 0; ic < C; ++ic)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < H; ++j)
                    for (int u = 0; u < r; ++u)
                        for (int v = 0; v < r; ++v)
                        {
                            hs[oc*H*H + i*H + j] += d[ic*(H+r-1)*(H+r-1) + (i+u)*(H+r-1) + j+v] 
                                                    * g[oc*C*r*r + ic*r*r + u*r + v];
                        }
}

// d是im2col之后的矩阵
// e.g. const int K = 36, C = 16, H = 32, r = 3, m = 4;
// const int K = 36, C = 16, H = 30, r = 3, m = 4;
// const int tile_h = ceil((float)H / m), tile_w = tile_h;
// const int P = tile_h * tile_w;
// const int alpha = m + r - 1;
void winograd_conv(float* d, float* g, float* h, int K, int C, int H, int r, int m)
{
    const int tile_h = ceil((float)H / m), tile_w = tile_h;
    const int P = tile_h * tile_w;
    const int alpha = m + r - 1;

    float* U[alpha][alpha];
    float* V[alpha][alpha];
    float* M[alpha][alpha];
    float* M1;

    for (int xi = 0; xi < alpha; ++xi)
        for (int mu = 0; mu < alpha; ++mu)
        {
            U[xi][mu] = (float*)malloc(K * C * sizeof(float));
            V[xi][mu] = (float*)malloc(C * P * sizeof(float));
            M[xi][mu] = (float*)malloc(K * P * sizeof(float));
        }
    M1 = (float*)malloc(alpha * alpha * sizeof(float));
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c)
        {
            float* u1 = (float*)malloc(alpha * r * sizeof(float));
            float* u = (float*)malloc(alpha * alpha * sizeof(float));
            basic_mm(G, g + k*C*r*r + c*r*r, u1, alpha, r, r, r, r);
            basic_mm(u1, GT, u, alpha, r, alpha, r, alpha);
            for (int xi = 0; xi < alpha; ++xi)
                for (int mu = 0; mu < alpha; ++mu)
                    U[xi][mu][k * C + c] = u[xi * alpha + mu];
            free(u1);
            free(u);
        }
    for (int c = 0; c < C; ++c)
        for (int b = 0; b < P; ++b)
        {
            float* v1 = (float*)malloc(alpha * alpha * sizeof(float));
            float* v = (float*)malloc(alpha * alpha * sizeof(float));
            basic_mm(BT, d + c*P*alpha*alpha + b*alpha*alpha, v1, alpha, alpha, alpha, alpha, alpha);
            basic_mm(v1, B, v, alpha, alpha, alpha, alpha, alpha);
            for (int xi = 0; xi < alpha; ++xi)
                for (int mu = 0; mu < alpha; ++mu)
                    V[xi][mu][c * P + b] = v[xi * alpha + mu];
            free(v1);
            free(v);
        }
    for (int xi = 0; xi < alpha; ++xi)
        for (int mu = 0; mu < alpha; ++mu)
        {
            // print_tensor("U:", U[xi][mu], K, C);
            // print_tensor("V:", V[xi][mu], C, P);
            basic_mm(U[xi][mu], V[xi][mu], M[xi][mu], K, C, P, C, P);
            // system("pause");
        }
    for (int k = 0; k < K; ++k)
        for (int b = 0; b < P; ++b)
        {
            for (int xi = 0; xi < alpha; ++xi)
                for (int mu = 0; mu < alpha; ++mu)
                    M1[xi * alpha + mu] = M[xi][mu][k * P + b];
            float* m1 = (float*)malloc(m * alpha * sizeof(float));
            float* m2 = (float*)malloc(m * m * sizeof(float));
            basic_mm(AT, M1, m1, m, alpha, alpha, alpha, alpha);
            basic_mm(m1, A, m2, m, alpha, m, alpha, m);
            // print_tensor("M1:", M1, alpha, alpha);
            // print_tensor("m1:", m1, m, alpha);
            // print_tensor("m2:", m2, m, m);
            for (int xm = 0; xm < m; ++xm)
                for (int ym = 0; ym < m; ++ym)
                {
                    int x_base = b / tile_w * m, y_base = b % tile_w * m;
                    if (x_base + xm < H && y_base + ym < H)
                        h[k * H * H + (x_base + xm) * H + (y_base + ym)] = m2[xm * m + ym];
                }
            free(m1);
            free(m2);
        }
    for (int xi = 0; xi < alpha; ++xi)
        for (int mu = 0; mu < alpha; ++mu)
        {
            free(U[xi][mu]);
            free(V[xi][mu]);
            free(M[xi][mu]);
        }
}

void im2col(float* in, float* out, int K, int C, int H, int r, int m)
{
    const int tile_h = ceil((float)H / m), tile_w = tile_h;
    const int P = tile_h * tile_w;
    const int alpha = m + r - 1;

    for (int c = 0; c < C; ++c)
        for (int x_tilde = 0; x_tilde < tile_h; ++x_tilde)
            for (int y_tilde = 0; y_tilde < tile_w; ++y_tilde)
                for (int j = 0; j < alpha*alpha; ++j)
                {
                    int b = x_tilde * tile_w + y_tilde;
                    if (x_tilde*m + j/alpha < H+r-1 && y_tilde*m + j%alpha < H+r-1)
                        out[c*P*alpha*alpha + b*alpha*alpha + j] = 
                                in[c*(H+r-1)*(H+r-1) + (x_tilde*m + j/alpha)*(H+r-1) + (y_tilde*m + j%alpha)];
                    else
                        out[c*P*alpha*alpha + b*alpha*alpha + j] = 0;
                }
}

