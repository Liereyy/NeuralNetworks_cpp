#pragma once
#include "tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"

namespace torch
{
    template <typename T>
    pair<size_t, size_t> max_index(tensor<T> t)
    {
        size_t index = 0;
        for (size_t i = 1; i < t.data_size(); ++i)
            if (t[i] > t[index])
                index = i;
        return make_pair<size_t, size_t>(index / t.dsize[1], index % t.dsize[1]);
    }

    template <typename T>
    tensor<T> soft_max(tensor<T> t)
    {
        double sum = 0;
        for (size_t i = 0; i < t.data_size(); ++i)
        {
            t[i] = pow(consts::e, t[i] > 18 ? 18 : t[i]);
            sum += t[i];
        }
        for (size_t i = 0; i < t.data_size(); ++i)
            t[i] = t[i] / sum;
        return t;
    }

    template <typename T>
    tensor<T> log(tensor<T> t)
    {
        for (size_t i = 0; i < t.data_size(); ++i)
            t[i] = std::log(t[i]);
        return t;
    }

    template <typename T>
    double sum(tensor<T>& t)
    {
        double res = 0;
        for (size_t i = 0; i < t.data_size(); ++i)
            res += t[i];
        return res;
    }
}