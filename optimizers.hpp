#pragma once
#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "basic_optimizer.hpp"
#include "basic_node.hpp"

namespace torch
{
    class GD : public basic_optimizer
    {
    public:
        float learning_rate;
    public:
        GD(grad_node<float>* r, float lr, size_t bs)
            : basic_optimizer(r, bs), learning_rate(lr)
        {}

        void update()
        {
            for (auto p : grad_node<float>::Gragh)
                if (p->ntype == node_type::_Variable && p->train_able)
                {
                    basic_tensor<float> gradient = this->average_grad(p);

                    p->reset_value();
                    *p->value = *p->value - learning_rate * gradient;
                }
        }
    };
}