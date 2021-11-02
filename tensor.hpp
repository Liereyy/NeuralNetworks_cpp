#pragma once
#include "basic_tensor.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"

namespace torch
{
    template <typename T>
    class tensor
    {
    public:
        typedef tensor<T>    self_type;
    public:
        basic_tensor<T>* _tensor;
        grad_node<T>* gnode;
    public:
        tensor() : _tensor(nullptr), gnode(nullptr) {}
        
        tensor(basic_tensor<T>* t)
            : _tensor(t)
        {
            gnode = new Variable<T>(_tensor, true);
        }

        ~tensor()
        {}

        template<size_t S>
        tensor(vector<T> v, const int (&psize)[S], TensorOptions _options=default_TensorOptions)
        {
             vector<int> tmp(psize, psize + S);
            _tensor = new basic_tensor<float>(v, tmp, _options);
            gnode = new Variable<T>(_tensor, _options._requires_grad);
        }

        template<typename V, size_t S>
        tensor(const V (&p)[S], TensorOptions _options=default_TensorOptions)
            : _tensor(new basic_tensor<T>(p, S, _options))
        {
            gnode = new Variable<T>(_tensor, _options._requires_grad);
        }

        template<size_t S1, size_t S2>
        tensor(const float (&pdata)[S1], const int (&psize)[S2], TensorOptions _options=default_TensorOptions)
        {
            vector<float> tmp1(pdata, pdata + S1);
            vector<int> tmp2(psize, psize + S2);
            _tensor = new basic_tensor<float>(tmp1, tmp2, _options);
            gnode = new Variable<T>(_tensor, _options._requires_grad);
        }

        tensor(basic_tensor<T>* bt, grad_node<T>* gn)
            : _tensor(bt), gnode(gn)
        {}

        tensor(const tensor<T>& t)
            : _tensor(t._tensor), gnode(t.gnode)
        {
            gnode->reset_value();
        }

        self_type& operator=(const tensor<T>& t)
        {
            _tensor = t._tensor;
            gnode = t.gnode;
            // gnode->value = _tensor;
            gnode->reset_value();
            return *this;
        }

        template <typename V>
        friend ostream& operator<<(ostream& os, const tensor<V>& t)
        {
            os << *(t._tensor);
            return os;
        }

        template<typename... Args>
        T& operator()(Args... i)
        {
            return (*_tensor)(i...);
        }

        T& operator[](size_t index)
        {
            return (*_tensor)[index];
        }

        vector<int> size()
        {
            return _tensor->size();
        }

        int data_size()
        {
            return _tensor->data_size();
        }

        size_t dimension()
        {
            return _tensor->dimension();
        }

        template<typename V, size_t S> 
        self_type reshape(const V (&p)[S])
        {
            auto t = *this;
            *(t._tensor) = t._tensor->reshape(p);
            return t;
        }

        self_type tr()  // 转置
        {
            self_type t = *this;
            *(t._tensor) = t._tensor->tr();
            return t;
        }
    };
}