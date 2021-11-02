#pragma once
#include "basic_tensor.hpp"
#include "tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"

namespace torch
{
    namespace F
    {
        template <typename T>
        tensor<T> Sigmoid(const tensor<T>& t1)
        {
            basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
            grad_node<T>* gnode = new Logistic<T>({t1.gnode}, new_bt);
            tensor<T> t(new_bt, gnode);
            return t;
        }

        template <typename T>
        tensor<T> relu(const tensor<T>& t1)
        {
            basic_tensor<T>* new_bt = new basic_tensor<T>(*t1._tensor);
            grad_node<T>* gnode = new ReLU<T>({t1.gnode}, new_bt);
            tensor<T> t(new_bt, gnode);
            return t;
        }
    }
}