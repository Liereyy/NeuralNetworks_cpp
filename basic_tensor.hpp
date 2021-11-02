#pragma once
#include <iostream>
#include <fstream>
#include <numeric>
#include "io_overloads.hpp"
#include "assist_functs.hpp"
#include <cmath>
using namespace std;

namespace torch
{
    struct TensorOptions
    {
        bool _requires_grad;

        explicit TensorOptions(bool rg=true)
            : _requires_grad(rg)
        {}
    };
    
    auto default_TensorOptions = TensorOptions();

    int total_size(const int* v, int n)
    {
        return accumulate(v, v + n, 1, multiplies<size_t>());
    }

    int total_size(const vector<int>& v)
    {
        return accumulate(v.begin(), v.end(), 1, multiplies<size_t>());
    }

    template <typename T>
    class basic_tensor
    {
    
    public:
        typedef basic_tensor<T>    self_type;
    public:
        // 均按一维向量存储，维度用dsize体现
        vector<int> dsize;
        vector<T> data;
        TensorOptions options;

    public:
        int index_of(const vector<int>& v) const
        {
            int index = v[0];
            for (size_t i = 1; i < v.size(); ++i)
                index = index * dsize[i] + v[i];
            return index;
        }

        void print(ostream& os, size_t d, vector<int> index)
        {
            if (d == dsize.size())
            {
                os << data[index_of(index)];
                return;
            }
            os << '[';
            for (int i = 0; i < dsize[d] - 1; ++i)
            {
                vector<int> v = index;
                v[d] = i;
                print(os, d + 1, v);
                os << ", ";
                if (d < dsize.size() - 1) os << '\n';
            }
            vector<int> v = index;
            v[d] = dsize[d] - 1;
            print(os, d + 1, v);
            os << "]";
        }
    public:
        basic_tensor()
            : options(default_TensorOptions)
        {}

        template<size_t S>
        basic_tensor(const int (&p)[S], TensorOptions _options=default_TensorOptions)  // 维度
            : data(vector<T>(total_size(p, S), 0)), dsize(vector<int>(p, p + S)), options(_options)
        {}

        template<typename V>
        basic_tensor(const V* p, int S, TensorOptions _options=default_TensorOptions)  // 维度
            : dsize(vector<int>(p, p + S)), options(_options)
        {
            data.reserve(total_size(p, S));
        }

        basic_tensor(const vector<T>& v, const vector<int>& d, TensorOptions _options=default_TensorOptions)
            : data(v), dsize(d), options(_options)
        {}

        template<size_t S>
        basic_tensor(const vector<T>& vdata, const int (&p)[S], TensorOptions _options=default_TensorOptions)
            : data(vdata), dsize(vector<int>(p, p + S)), options(_options)
        {}

        basic_tensor(const basic_tensor<T>& bt)
            : dsize(bt.dsize), data(bt.data), options(bt.options)
        {}

    public:
        template <typename V>
        friend ostream& operator<<(ostream& os, basic_tensor<V> t)
        {
            os << "value=\n";
            vector<int> v(t.dimension(), 0);
            t.print(os, 0, v);
            os << "\nsize=" << t.dsize << endl;
            return os;
        }

        self_type operator-()
        {
            self_type tmp = *this;
            for (size_t i = 0; i < tmp.data.size(); ++i)
                tmp.data[i] = -tmp.data[i];
            return tmp;
        }

        template<typename... Args>
        T& operator()(Args... i)
        {
            return data[index_of({i...})];
        }

        T& operator[](size_t index)
        {
            return data[index];
        }

        vector<int> size()
        {
            return dsize;
        }

        int data_size()
        {
            return data.size();
        }

        size_t dimension()
        {
            return dsize.size();
        }

        template<typename V, size_t S> 
        basic_tensor<T> reshape(const V (&p)[S])  // 编译期推断参数大小S
        {
            if (total_size(p, S) != total_size(dsize))
            {
                cerr << "reshape failed.\n";
                exit(0);
            }
            auto t = *this;
            t.dsize = vector<int>(p, p + S);
            return t;
        }

        basic_tensor<T> reshape(vector<int> _dsize)
        {
            if (total_size(_dsize) != total_size(dsize))
            {
                cerr << "reshape failed.\n";
                exit(0);
            }
            auto t = *this;
            t.dsize = _dsize;
            return t;
        }

        self_type tr()  // 转置
        {
            self_type t = *this;
            swap(t.dsize.back(), t.dsize[t.dsize.size() - 2]);
            pair<int, int> p_matrix = last2_of_v(dsize);
            size_t h = p_matrix.first, w = p_matrix.second;
            size_t matrix_size = h * w;
            for (size_t p = 0; p < t.data.size() / matrix_size; ++p)
                for (size_t i = 0; i < w; ++i)
                    for (size_t j = 0; j < h; ++j)
                        t.data[p * matrix_size + i * h + j] = data[p * matrix_size + j * w + i];
            return t;
        }

        self_type flatten()  // 展平
        {
            self_type t = *this;
            t.dsize = vector<int>(1, dimension());
            return t;
        }
    };
}