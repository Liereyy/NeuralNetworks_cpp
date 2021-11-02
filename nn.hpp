#pragma once
#include "tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"

namespace torch
{
    namespace nn
    {
        template <typename T>
        struct Conv2d
        {
            int in_channels;
            int out_channels;
            pair<int, int> kernel_size;
            pair<int, int> stride;
            tensor<T> kernel;
            tensor<T> bias;

            Conv2d(int _in_channels, int _out_channels, pair<int, int> ks, pair<int, int> _stride)
                : in_channels(_in_channels), out_channels(_out_channels), kernel_size(ks), stride(_stride)
            {
                basic_tensor<T>* kernel_value = new basic_tensor<T>(brandn({out_channels, in_channels, ks.first, ks.second}) / 5);
                basic_tensor<T>* bias_value = new basic_tensor<T>(brandn({in_channels, 1}) / 5);
                kernel = tensor<T>(kernel_value);  // 新建variable节点
                bias = tensor<T>(bias_value);  // 新建variable节点
                // kernel和bias的gnode按默认设置为_Variable类型
            }

            tensor<T> operator()(const tensor<T>& t1)
            {
                basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
                grad_node<T>* gnode = new Conv2<T>({t1.gnode, kernel.gnode, bias.gnode}, 
                                                        in_channels, out_channels, new_bt, kernel_size, stride);
                tensor<T> t(new_bt, gnode);
                return t;
            }
        };

        template <typename T>
        struct Linear
        {
            int in_features;
            int out_features;
            tensor<T> A;  // A右乘

            Linear(int _in_features, int _out_features)
                : in_features(_in_features), out_features(_out_features)
            {
                A._tensor = new basic_tensor<T>(brandn({in_features, out_features}) / 5);
                A.gnode = new Variable<T>(A._tensor, true);
            }

            tensor<T> operator()(const tensor<T>& t1)
            {
                basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
                grad_node<T>* gnode = new mm<T>({t1.gnode, A.gnode}, new_bt);
                tensor<T> t(new_bt, gnode);
                return t;
            }
        };

        template <typename T>
        struct MaxPool2d
        {
            int channels;
            pair<int, int> kernel_size;

            MaxPool2d(pair<int, int> ks)
                : kernel_size(ks)
            {}

            tensor<T> operator()(const tensor<T>& t1)
            {
                this->channels = t1._tensor->dsize[0];
                basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
                grad_node<T>* gnode = new MaxPool2<T>({t1.gnode}, channels, new_bt, kernel_size);
                tensor<T> t(new_bt, gnode);
                return t;
            }
        };

        template <typename T>
        struct Reshape
        {
            vector<int> target_dsize;

            Reshape(const vector<int>& _dsize)
                : target_dsize(_dsize)
            {}

            tensor<T> operator()(const tensor<T>& t1)
            {
                basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
                grad_node<T>* gnode = new _reshape<T>({t1.gnode}, new_bt, target_dsize);
                tensor<T> t(new_bt, gnode);
                return t;
            }
        };

        template <typename T>
        struct CrossEntropyLoss
        {
            CrossEntropyLoss() {}

            tensor<T> operator()(const tensor<T>& t1, const tensor<T>& t2)
            {
                basic_tensor<T>* new_bt = new basic_tensor<T>({1, 1});
                grad_node<T>* gnode = new CrossEntropy<T>({t1.gnode, t2.gnode}, new_bt);
                tensor<T> t(new_bt, gnode);
                return t;
            }
        };
    }
}