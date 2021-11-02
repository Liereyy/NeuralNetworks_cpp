#pragma once
#include "tensor.hpp"
#include <vector>
#include "constants.hpp"

namespace torch
{
    template<size_t S>
    tensor<float> zeros(const int (&p)[S])
    {
        auto tmp_data = vector<float>(total_size(p, S), 0);
        auto tmp_size = vector<int>(p, p + S);
        auto _tensor = new basic_tensor<float>(tmp_data, tmp_size);
        auto t = tensor<float>(_tensor);
        return t;
    }

    tensor<float> eye(int size, int)
    {
        auto tmp_data = vector<float>(size * size, 0);
        auto tmp_size = vector<int>({size, size});
        auto _tensor = new basic_tensor<float>(tmp_data, tmp_size);
        auto t = tensor<float>(_tensor);
        for (int i = 0; i < size; ++i)
            t._tensor->data[i * size + i] = 1;
        return t;
    }

    template<size_t S> 
    tensor<float> ones(const int (&p)[S])
    {
        auto tmp_data = vector<float>(total_size(p, S), 1);
        auto tmp_size = vector<int>(p, p + S);
        auto _tensor = new basic_tensor<float>(tmp_data, tmp_size);
        auto t = tensor<float>(_tensor);
        return t;
    }

    template<size_t S> 
    tensor<float> rand(const int (&p)[S])
    {
        auto tmp_data = vector<float>();
        auto tmp_size = vector<int>(p, p + S);
        auto _tensor = new basic_tensor<float>(tmp_data, tmp_size);
        auto t = tensor<float>(_tensor);
        t._tensor = new basic_tensor<float>(tmp_data, tmp_size);
        for (size_t i = 0; i < total_size(p, S); ++i)
            t._tensor->data.push_back(pow(-1, std::rand()) * std::rand() / float(RAND_MAX));
        return t;
    }

    template<size_t S> 
    tensor<float> randn(const int (&p)[S])
    {
        auto tmp_data = vector<float>();
        auto tmp_size = vector<int>(p, p + S);
        auto _tensor = new basic_tensor<float>(tmp_data, tmp_size);
        auto t = tensor<float>(_tensor);
        for (int i = 0; i < total_size(p, S); ++i)
        {
            float u1 = std::rand() / float(RAND_MAX), u2 = std::rand() / float(RAND_MAX);
            t._tensor->data.push_back(sqrt(-2 * std::log(u1)) * cos(2 * consts::PI * u2));  // Box-Muller
        }
        return t;
    }
}