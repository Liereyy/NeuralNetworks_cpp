#pragma once
#include "tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"
#include <windows.h>
using namespace std;

LARGE_INTEGER cpuFreq_to;
LARGE_INTEGER startTime_to;
LARGE_INTEGER endTime_to;
double runTime_to;

namespace torch
{
    template <typename T>
    tensor<T> operator+(const tensor<T>& t1, const tensor<T>& t2)
    {
        basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
        grad_node<T>* gnode = new Add<T>({t1.gnode, t2.gnode}, new_bt);
        tensor<T> t(new_bt, gnode);
        for (int i = 0; i < t1._tensor->data_size(); ++i)
            t._tensor->data[i] += t2._tensor->data[i];
        return t;
    }

    template <typename T>
    tensor<T> operator*(const tensor<T>& t1, const tensor<T>& t2)
    {
        if (t1._tensor->dsize.size() == 2)  // 二维
        {
            int m = t1._tensor->dsize[0], n = t1._tensor->dsize[1], k = t2._tensor->dsize[1];
            if (n != t2._tensor->dsize[0])
            {
                cerr << "mm fail.\n";
                exit(0);
            }
            basic_tensor<T>* new_bt = new basic_tensor<T>(bzeros({m, k}));
            grad_node<T>* gnode = new mm<T>({t1.gnode, t2.gnode}, new_bt);
            tensor<T> t(new_bt, gnode);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    for (int p = 0; p < n; ++p)
                        (*new_bt)[i * k + j] += (*t1._tensor)[i * n + p] * (*t2._tensor)[p * k + j];
            return t;
        }
        else if (t1._tensor->dsize.size() == 3)  // 三维
        {
            int batches = t1._tensor->dsize[0];
            int m = t1._tensor->dsize[1], n = t1._tensor->dsize[2], k = t2._tensor->dsize[2];
            assert(n == t2._tensor->dsize[1]);

            basic_tensor<T>* new_bt = new basic_tensor<T>(bzeros({batches, m, k}));
            grad_node<T>* gnode = new mm<T>({t1.gnode, t2.gnode}, new_bt);
            tensor<T> t(new_bt, gnode);
            for (int b = 0; b < batches; ++b)
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < k; ++j)
                        for (int p = 0; p < n; ++p)
                            (*new_bt)(b, i, j) += (*t1._tensor)(b, i, p) * (*t2._tensor)(b, p, j);
            return t;
        }
        exit(0);
    }

    // template <typename T>
    // tensor<T> CrossEntropyLoss(const tensor<T>& t1, const tensor<T>& t2)
    // {
    //     basic_tensor<T>* new_bt = new basic_tensor<T>({1, 1});
    //     grad_node<T>* gnode = new CrossEntropy<T>({t1.gnode, t2.gnode}, new_bt);
    //     tensor<T> t(new_bt, gnode);
    //     return t;
    // }

    

    // template <typename T>
    // tensor<T> Conv2d(const tensor<T>& t1, int _in_channels, int _out_channels, pair<int, int> ks, pair<int, int> _stride)
    // {
    //     basic_tensor<T>* kernel_value = new basic_tensor<T>(brandn({_out_channels, _in_channels, ks.first, ks.second}) / 5);
    //     basic_tensor<T>* bias_value = new basic_tensor<T>(brandn({1, 1}) / 5);
    //     tensor<T> kernel(kernel_value);  // 新建variable节点
    //     tensor<T> bias(bias_value);  // 新建variable节点
    //     basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
    //     grad_node<T>* gnode = new Conv2<T>({t1.gnode, kernel.gnode, bias.gnode}, 
    //                                             _in_channels, _out_channels, new_bt, ks, _stride);
    //     tensor<T> t(new_bt, gnode);
    //     return t;
    // }

    // template <typename T>
    // tensor<T> MaxPool2d(const tensor<T>& t1, int _in_channels, pair<int, int> ks)
    // {
    //     basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
    //     grad_node<T>* gnode = new MaxPool2<T>({t1.gnode}, _in_channels, new_bt, ks);
    //     tensor<T> t(new_bt, gnode);
    //     return t;
    // }

    // template <typename T>
    // tensor<T> Reshape(const tensor<T>& t1, vector<int> _dsize)
    // {
    //     basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
    //     grad_node<T>* gnode = new _reshape<T>({t1.gnode}, new_bt, _dsize);
    //     tensor<T> t(new_bt, gnode);
    //     return t;
    // }
}