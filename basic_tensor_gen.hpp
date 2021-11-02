#pragma once
#include "basic_tensor.hpp"
#include <vector>
#include "constants.hpp"

namespace torch
{
    template<size_t S>
    basic_tensor<float> bzeros(const int (&p)[S])
    {
        auto t = basic_tensor<float>();
        t.dsize = vector<int>(p, p + S);
        t.data = vector<float>(total_size(p, S), 0);
        return t;
    }

    basic_tensor<float> beye(int size, int)
    {
        auto t = basic_tensor<float>();
        t.dsize = {size, size};
        t.data = vector<float>(size * size, 0);
        for (int i = 0; i < size; ++i)
            t.data[i * size + i] = 1;
        return t;
    }

    template<size_t S> 
    basic_tensor<float> bones(const int (&p)[S])
    {
        auto t = basic_tensor<float>();
        t.dsize = vector<int>(p, p + S);
        t.data = vector<float>(total_size(p, S), 1);
        return t;
    }

    template<size_t S> 
    basic_tensor<float> brand(const int (&p)[S])
    {
        auto t = basic_tensor<float>();
        t.dsize = vector<int>(p, p + S);
        for (size_t i = 0; i < total_size(p, S); ++i)
            t.data.push_back(pow(-1, std::rand()) * std::rand() / float(RAND_MAX));
        return t;
    }

    template<size_t S> 
    basic_tensor<float> brandn(const int (&p)[S])
    {
        auto t = basic_tensor<float>();
        t.dsize = vector<int>(p, p + S);
        for (int i = 0; i < total_size(p, S); ++i)
        {
            float u1 = std::rand() / float(RAND_MAX), u2 = std::rand() / float(RAND_MAX);
            t.data.push_back(sqrt(-2 * log(u1)) * cos(2 * consts::PI * u2));  // Box-Muller
        }
        return t;
    }
}