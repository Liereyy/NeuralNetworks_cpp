#pragma once
#include <iostream>
#include <vector>
#include "basic_tensor.hpp"
#include "basic_tensor_gen.hpp"
#include <windows.h>
using namespace std;

LARGE_INTEGER cpuFreq_bn;
LARGE_INTEGER startTime_bn;
LARGE_INTEGER endTime_bn;
double runTime_bn;

// #define TIME_COMPUTE_JACOBI

namespace torch
{
    enum node_type{_Variable, _Add, _mm, _Logistic, _ReLU, _SoftMax, _CrossEntropy,
                        _Conv2, _MaxPool2, _Linear, _Reshape};

    template <typename T>
    class grad_node
    {
    public:
        vector<grad_node<T>*> parents;
        vector<grad_node<T>*> children;
        basic_tensor<T>* value;  // 此处value用指针，与包装其的tensor的_tensor指向一个空间
        basic_tensor<T> jacobi;  // 结果节点对本节点的jacobi矩阵
        bool bcomputed;
        bool bjacobi;

        // for Conv2d and MaxPool2d
        int in_channels;
        int out_channels;
        basic_tensor<T> kernel;
        pair<int, int> kernel_size;
        basic_tensor<T> bias;
        pair<int, int> stride;

        // for Linear, y = xA^T + b
        basic_tensor<T> LA;
        int in_features;
        int out_features;
        basic_tensor<T> Lb;

        // for reshape
        vector<int> reshape_dsize;

        static vector<grad_node<T>*> Gragh;

        void reset_all_jacobi()
        {
            for (auto p : Gragh)
                p->reset_jacobi();
        }

        node_type ntype;  // 标识节点类别
        bool train_able;

        grad_node(vector<grad_node<T>*> vparents, basic_tensor<T>* data, node_type nid=_Variable)
            : parents(vparents), value(data), bcomputed(false), bjacobi(false),
                    ntype(nid), train_able(true)
        {
            Gragh.push_back(this);
            for (auto p : vparents)
                p->children.push_back(this);
        }

        virtual ~grad_node()
        {

        }

        void forward()
        {
            for (auto p : parents)
                if (!p->bcomputed)  // 父节点值未计算
                    p->forward();
            if (!bcomputed)
                compute();
        }

        virtual void compute() = 0;

        basic_tensor<T> backward(grad_node<T>* result)
        {
            if (!bjacobi)  // jacobi未计算
            {
                if (result == this)
                    jacobi = beye(value->data_size(), value->data_size());
                else
                {
                    jacobi = bzeros({result->value->data_size(), value->data_size()});
                    for (auto c : children)
                    {
                        if (c->bcomputed)
                        {
                            QueryPerformanceFrequency(&cpuFreq_bn);
                            QueryPerformanceCounter(&startTime_bn);
                            
                            auto y = c->compute_jacobi(this);

                            QueryPerformanceCounter(&endTime_bn);
                            runTime_bn = (((endTime_bn.QuadPart - startTime_bn.QuadPart) * 1000.0f) / cpuFreq_bn.QuadPart);
                            #ifdef TIME_COMPUTE_JACOBI
                            cout << "compute_jacobi() time: " << runTime_bn << "ms. " << this->value->dsize << c->value->dsize << endl;
                            #endif
                            // cout << "basic_node.hpp: " << c->value->dsize << ' ' << this->value->dsize << ' ' << 
                            //             y.use_mirror << ' ' << y.dsize << ' ' << y.map_dsize << endl;
                            jacobi = jacobi + c->backward(result) * y;
                        }
                    }
                }
                bjacobi = true;
            }
            return jacobi;
        }

        // 计算parent节点对本节点的jacobi矩阵
        virtual basic_tensor<T> compute_jacobi(grad_node<T>* parent) = 0;

        void reset_jacobi()
        {
            bjacobi = false;
            fill(jacobi.data.begin(), jacobi.data.end(), 0);
        }

        void reset_value(bool reset_children=true)
        {
            bcomputed = false;
            bjacobi = false;
            if (reset_children)
                for (auto c : children)
                    c->reset_value();
        }
    };

    template<typename T>
    vector<grad_node<T>* > grad_node<T>::Gragh;
}
