#pragma once
#include "basic_tensor.hpp"
#include "basic_node.hpp"
#include "basic_tensor_gen.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_tensor_operator.hpp"
#include <assert.h>
#include <windows.h>
#include "winograd_conv.hpp"
using namespace std;

// #define TIME_MM_NODE
// #define TIME_CONV2D_NODE
// #define TIME_CONV2D_JACOBI

#if (defined TIME_MM_NODE) || (defined TIME_CONV2D_NODE) || (defined TIME_CONV2D_JACOBI)
LARGE_INTEGER cpuFreq_en;
LARGE_INTEGER startTime_en;
LARGE_INTEGER endTime_en;
double runTime_en;
LARGE_INTEGER startTime_en1;
LARGE_INTEGER endTime_en1;
double runTime_en1;
#endif

namespace torch
{
    // 变量节点，叶子
    template<typename T>
    class Variable : public grad_node<T>
    {
    public:
        pair<int, int> dim;
    public:
        Variable(basic_tensor<T>* data, bool t=true)
            : grad_node<T>({}, data, node_type::_Variable)
        {
            this->train_able = t;
        }

        void compute()
        {}

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            return beye(this->value->data_size(), this->value->data_size());
        }
    };

    // 加法
    template <typename T>
    class Add : public grad_node<T>
    {
    public:
        Add() {}
        Add(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_Add)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            *this->value = *this->parents[0]->value + *this->parents[1]->value;
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            return beye(this->value->data_size(), this->value->data_size());
        }
    };

    // 矩阵相乘
    template <typename T>
    class mm : public grad_node<T>
    {
    public:
        mm(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_mm)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            #ifdef TIME_MM_NODE
            QueryPerformanceFrequency(&cpuFreq_en);
            QueryPerformanceCounter(&startTime_en);
            #endif

            this->bcomputed = true;
            *this->value = *this->parents[0]->value * *this->parents[1]->value;

            #ifdef TIME_MM_NODE
            QueryPerformanceCounter(&endTime_en);
            runTime_en = (((endTime_en.QuadPart - startTime_en.QuadPart) * 1000.0f) / cpuFreq_en.QuadPart);
            cout << "mm: " << this->parents[0]->value->dsize << this->parents[1]->value->dsize
                       << "compute consume = " << runTime_en << " ms" << endl;
            #endif
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            //    A    *    B    ->    C
            // (m x n) * (n x k) -> (m x k)
            int m = this->value->dsize[0], k = this->value->dsize[1], n;
            basic_tensor<T> J = bzeros({this->value->data_size(), parent->value->data_size()});
            if (parent == this->parents[0])
            {
                // dC/dA
                // jacobi: (m x k) * (m x n)
                // i = 0...m-1, j = 0...k-1, p = 0...n-1
                // J(i * k + j, i * n + p) = B(p, j)
                n = parent->value->dsize[1];
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < k; ++j)
                        for (int p = 0; p < n; ++p)
                            J(i * k + j, i * n + p) = (*this->parents[1]->value)(p, j);
            }
            else if (parent == this->parents[1])
            {
                // dC/dB
                // jacobi: (m x k) * (n x k)
                // i = 0...m-1, j = 0...k-1, p = 0...n-1
                // J(i * k + j, p * k + j) = A(i, p)
                n = parent->value->dsize[0];
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < k; ++j)
                        for (int p = 0; p < n; ++p)
                            J(i * k + j, p * k + j) = (*this->parents[0]->value)(i, p);
            }
            else
                cerr << "mm jacobi fail.\n";
            return J;
        }
    };

    // logistic
    template <typename T>
    class Logistic : public grad_node<T>
    {
    public:
        Logistic(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_Logistic)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            *this->value = *this->parents[0]->value;
            for (int i = 0; i < this->value->data_size(); ++i)
                (*this->value)[i] = 1.0 / (1.0 + pow(consts::e, -(*this->value)[i]));
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            auto res = beye(this->value->data_size(), this->value->data_size());
            for (int i = 0; i < res.data_size(); ++i)
                if (res[i] > 1e-2)
                    res[i] = (1 - (*this->value)[i]) * (*this->value)[i];
            return res;
        }
    };

    // ReLU
    template <typename T>
    class ReLU : public grad_node<T>
    {
    public:
        ReLU(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_ReLU)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            *this->value = *this->parents[0]->value;
            for (int i = 0; i < this->value->data_size(); ++i)
                (*this->value)[i] = (*this->value)[i] >= 0 ? (*this->value)[i] : 0;
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            auto res = beye(this->value->data_size(), this->value->data_size());
            for (int i = 0; i < this->value->data_size(); ++i)
                res[i * this->value->data_size() + i] = (*this->value)[i] > 0 ? 1 : 0;
            return res;
        }
    };

    // SoftMax
    template <typename T>
    class SoftMax : public grad_node<T>
    {
    public:
        SoftMax(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_SoftMax)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            *this->value = soft_max(*this->parents[0]->value);
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            // 此函数不进行计算，放在求交叉熵再计算
            return beye(this->value->data_size(), this->value->data_size());
        }
    };

    // 交叉熵
    template <typename T>
    class CrossEntropy : public grad_node<T>
    {
    public:
        CrossEntropy(vector<grad_node<T>*> vparents, basic_tensor<T>* data)
            : grad_node<T>(vparents, data, node_type::_CrossEntropy)
        {
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            // 第一个父节点进行SoftMax，第二个父节点为标签One-Hot向量，再计算交叉熵
            this->bcomputed = true;
            *this->value = basic_tensor<T>({1, 1});  // 空标量

            basic_tensor<T> prob = soft_max(*this->parents[0]->value);
            size_t label_index = 0;
            while ((*this->parents[1]->value)[label_index] == 0) ++label_index;
            (*this->value)(0, 0) = -std::log(prob[label_index]);
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            basic_tensor<T> res;
            basic_tensor<T> prob = soft_max(*this->parents[0]->value);
            if (parent == this->parents[0])
                res = prob - *this->parents[1]->value;
            else
                res = -torch::log(prob);
            return res;
        }
    };

    template <typename T>
    class Conv2 : public grad_node<T>
    {
    public:
        Conv2(vector<grad_node<T>*> vparents, int _in_channels, int _out_channels, basic_tensor<T>* data,
                pair<int, int> ks, pair<int, int> _stride=pair<int, int>(1, 1))
            : grad_node<T>(vparents, data, node_type::_Conv2)
        {
            this->in_channels = _in_channels;
            this->out_channels = _out_channels;
            this->kernel_size = ks;
            this->stride = _stride;
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            #ifdef TIME_CONV2D_NODE
            QueryPerformanceFrequency(&cpuFreq_en);
            QueryPerformanceCounter(&startTime_en);
            #endif

            // parents[0]:data_in, [1]:kernel, [2]:bias
            this->bcomputed = true;
            // (m, n) * (r, s) -> (m - r + 1, n - s + 1)
            int m = this->parents[0]->value->dsize[1];
            int n = this->parents[0]->value->dsize[2];
            int r = this->kernel_size.first;
            int s = this->kernel_size.second;
            auto& data_in = *this->parents[0]->value;
            auto& kernel = *this->parents[1]->value;
            auto& bias = *this->parents[2]->value;
            *this->value = bzeros({this->out_channels, m - r + 1, n - s + 1});
            auto value1 = bzeros({this->out_channels, m - r + 1, n - s + 1});

            float* d = (float*)&data_in.data[0];
            float* g = (float*)&kernel.data[0];
            float* h = (float*)&this->value->data[0];

            // // F(4x4, 3x3)
            const int K = this->out_channels, C = this->in_channels, H = this->value->dsize[1];
            int alpha = data_in.dsize[1];
            const int tile_h = ceil((float)H / 3), tile_w = tile_h;
            const int P = tile_h * tile_w;
            float* c = (float*)malloc(this->in_channels*P*alpha*alpha*sizeof(float));

            im2col(d, c, K, C, H, 3, 4);
            winograd_conv(c, g, (float*)&value1.data[0], K, C, H, 3, 4);

            #ifdef TIME_CONV2D_NODE
            QueryPerformanceCounter(&endTime_en);
            runTime_en = (((endTime_en.QuadPart - startTime_en.QuadPart) * 1000.0f) / cpuFreq_en.QuadPart);
            cout << "conv2d(winograd):  " << this->parents[0]->value->dsize << ' ' << this->parents[1]->value->dsize
                       << "  compute consume = " << runTime_en << " ms" << endl;
            #endif

            // 与标准卷积比较确保正确
            #ifdef TIME_CONV2D_NODE
            QueryPerformanceFrequency(&cpuFreq_en);
            QueryPerformanceCounter(&startTime_en1);
            #endif
            ordinary_conv(d, g, h, K, C, H, r, m);
            // for (int oc = 0; oc < this->out_channels; ++oc)
            //     for (int ic = 0; ic < this->in_channels; ++ic)
            //         for (int i = 0; i < m - r + 1; ++i)
            //             for (int j = 0; j < n - s + 1; ++j)
            //             {
            //                 for (int u = 0; u < this->kernel_size.first; ++u)
            //                     for (int v = 0; v < this->kernel_size.second; ++v)
            //                     {
            //                         value1(oc, i, j)
            //                             += kernel(oc, ic, u, v) * data_in(ic, i + u, j + v);
            //                     }
            //                 // value1(oc, i, j) += bias(oc, 0);
            //             }
            
            // for (int i = 0; i < this->value->data_size(); ++i)
            //     if (fabs((*this->value)[i] - value1[i]) > 1e-2)
            //     {
            //         cout << i << ": " << (*this->value)[i] << ' ' << value1[i] << endl;
            //         cout << "test fail.\n";
            //         exit(0);
            //     }

            #ifdef TIME_CONV2D_NODE
            QueryPerformanceCounter(&endTime_en1);
            runTime_en = (((endTime_en1.QuadPart - startTime_en1.QuadPart) * 1000.0f) / cpuFreq_en.QuadPart);
            cout << "conv2d:  " << this->parents[0]->value->dsize << ' ' << this->parents[1]->value->dsize
                       << "  compute consume = " << runTime_en << " ms" << endl;
            #endif
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            // if (this->bjacobi)
            //     return this->jacobi;
            // parents[0]:X, [1]:kernel, [2]:bias
            auto& data_in = *this->parents[0]->value;
            auto& kernel = *this->parents[1]->value;
            int m = this->parents[0]->value->dsize[1];
            int n = this->parents[0]->value->dsize[2];
            int r = this->kernel_size.first;
            int s = this->kernel_size.second;
            if (parent == this->parents[0])  // X雅可比
            {
                basic_tensor<T> J = bzeros({this->out_channels * (m - r + 1) * (n - s + 1), 
                                                this->in_channels * m * n});
                float* pJ = (float*)&J.data[0];
                float* pK = (float*)&kernel.data[0];
                for (int oc = 0; oc < this->out_channels; ++oc)
                    for (int ic = 0; ic < this->in_channels; ++ic)
                        for (int i = 0; i < m - r + 1; ++i)
                            for (int j = 0; j < n - s + 1; ++j)
                                for (int p = 0; p < m; ++p)
                                    for (int q = 0; q < n; ++q)
                                        if (p - i >= 0 && p - i < r && q - j >= 0 && q - j < s)
                                            *(pJ + (oc * (m - r + 1) * (n - s + 1) + i * (n - s + 1) + j)
                                                * (this->in_channels * m * n) + ic * m * n + p * n + q)
                                                // = kernel(oc, ic, p - i, q - j);
                                                = *(pK + oc * this->in_channels * r * s
                                                    + ic * r * s + (p - i) * s + q - j);
                return J;
            }
            else if (parent == this->parents[1])  // kernel雅可比
            {
                basic_tensor<T> J = bzeros({this->out_channels * (m - r + 1) * (n - s + 1), 
                                                this->out_channels * this->in_channels * r * s});
                float* pJ = (float*)&J.data[0];
                float* pD = (float*)&this->value->data[0];
                #ifdef TIME_CONV2D_JACOBI
                QueryPerformanceFrequency(&cpuFreq_en);
                QueryPerformanceCounter(&startTime_en);
                #endif

                for (int oc = 0; oc < this->out_channels; ++oc)
                    for (int ic = 0; ic < this->in_channels; ++ic)
                        for (int i = 0; i < m - r + 1; ++i)
                            for (int j = 0; j < n - s + 1; ++j)
                                for (int u = 0; u < r; ++u)
                                    for (int v = 0; v < s; ++v)
                                        *(pJ + (oc * (m - r + 1) * (n - s + 1) + i * (n - s + 1) + j) 
                                               * this->out_channels * this->in_channels * r * s
                                               + oc * this->in_channels * r * s + ic * r * s + u * s + v) 
                                            // = data_in(ic, i + u, j + v);
                                            = *(pD + ic * m * n + (i + u) * n + j + v);

                #ifdef TIME_CONV2D_JACOBI
                QueryPerformanceCounter(&endTime_en);
                runTime_en = (((endTime_en.QuadPart - startTime_en.QuadPart) * 1000.0f) / cpuFreq_en.QuadPart);
                cout << "conv2d_jacobi:  " << this->parents[0]->value->dsize << ' ' << this->parents[1]->value->dsize
                        << "  compute consume = " << runTime_en << " ms" << endl;
                #endif    
                
                return J;
            }
            else if (parent == this->parents[2])  // bias雅可比
            {
                basic_tensor<T> J({this->value->data_size(), this->out_channels});
                for (int oc = 0; oc < this->out_channels; ++oc)
                {
                    for (int i = 0; i < this->value->dsize[1]; ++i)
                        for (int j = 0; j < this->value->dsize[2]; ++j)
                            J(i * this->value->dsize[2] + j, oc) = 1;
                }
                return J;
            }
            cerr << "conv: jacobi fail.\n";
            exit(0);
        }
    };

    template <typename T>
    class MaxPool2 : public grad_node<T>
    {
    public:
        MaxPool2(vector<grad_node<T>*> vparents, int _in_channels, basic_tensor<T>* data, pair<int, int> ks)
            : grad_node<T>(vparents, data, node_type::_MaxPool2)
        {
            this->kernel_size = ks;  // 此处kernel为放缩比
            this->in_channels = _in_channels;
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            int h = this->parents[0]->value->dsize[1];
            int w = this->parents[0]->value->dsize[2];
            assert(h % this->kernel_size.first == 0 && w % this->kernel_size.second == 0);
            
            int nh = h / this->kernel_size.first, nw = w / this->kernel_size.second;
            *this->value = basic_tensor<T>({this->in_channels, nh, nw});

            for (int ic = 0; ic < this->in_channels; ++ic)
                for (int i = 0; i < nh; ++i)
                    for (int j = 0; j < nw; ++j)
                    {
                        int x_base = i * this->kernel_size.first;
                        int y_base = j * this->kernel_size.second;
                        T m = (*this->parents[0]->value)(ic, x_base, y_base);

                        for (int u = 0; u < this->kernel_size.first; ++u)
                            for (int v = 0; v < this->kernel_size.second; ++v)
                                if ((*this->parents[0]->value)(ic, x_base + u, y_base + v) > m)
                                    m = (*this->parents[0]->value)(ic, x_base + u, y_base + v);
                        (*this->value)(ic, i, j) = m;
                    }
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            basic_tensor<T> res = bzeros({this->value->data_size(), this->parents[0]->value->data_size()});
            for (int ic = 0; ic < this->in_channels; ++ic)
                for (int i = 0; i < this->value->dsize[1]; ++i)
                    for (int j = 0; j < this->value->dsize[2]; ++j)
                        res(ic * this->value->dsize[1] * this->value->dsize[2] + i * this->value->dsize[1] + j, 
                                ic * this->parents[0]->value->dsize[1] * this->parents[0]->value->dsize[2] + i * this->kernel_size.first * this->parents[0]->value->dsize[1] 
                                        + j * this->kernel_size.second) = 1;
            return res;
        }
    };


    template <typename T>
    class _reshape : public grad_node<T>
    {
    public:
        _reshape(vector<grad_node<T>*> vparents, basic_tensor<T>* data, vector<int>& _dsize)
            : grad_node<T>(vparents, data, node_type::_Reshape)
        {
            this->reshape_dsize = _dsize;
            compute();
            this->bcomputed = false;
        }

        void compute()
        {
            this->bcomputed = true;
            *this->value = *this->parents[0]->value;
            this->value->dsize = this->reshape_dsize;
        }

        basic_tensor<T> compute_jacobi(grad_node<T>* parent)
        {
            return beye(this->value->data_size(), 1);
        }
    };
}
