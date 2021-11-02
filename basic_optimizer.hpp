#pragma once
#include "tensor.hpp"
#include <cmath>
#include "constants.hpp"
#include "basic_node.hpp"
#include "extra_nodes.hpp"
#include <unordered_map>
#include <algorithm>
#include <windows.h>

LARGE_INTEGER cpuFreq_bo;
LARGE_INTEGER startTime_bo;
LARGE_INTEGER endTime_bo;
double runTime_bo;

using namespace std;
// #define TIME_FORWARD
// #define TIME_BACKWARD

namespace torch
{
    class basic_optimizer
    {
    public:
        size_t batch_size;
        grad_node<float>* result;
        unordered_map<grad_node<float>*, basic_tensor<float> > acc_grads;
        size_t step_no;
    public:
        basic_optimizer(grad_node<float>* r, size_t bs)
            : batch_size(bs), result(r)
        {}

        void forward_backward()
        {
            result->reset_all_jacobi();
            QueryPerformanceFrequency(&cpuFreq_bo);
            QueryPerformanceCounter(&startTime_bo);

            result->forward();

            QueryPerformanceCounter(&endTime_bo);
            runTime_bo = (((endTime_bo.QuadPart - startTime_bo.QuadPart) * 1000.0f) / cpuFreq_bo.QuadPart);
            #ifdef TIME_FORWARD
            cout << "basic_optimizer.hpp: forward() time: " << runTime_bo << "ms.\n";
            #endif


            QueryPerformanceCounter(&startTime_bo);
            
            for (auto p : grad_node<float>::Gragh)
                if (p->ntype == node_type::_Variable && p->train_able)
                {
                    p->backward(result);
                    basic_tensor<float> gradient = p->jacobi.reshape(p->value->dsize);

                    if (acc_grads.find(p) == acc_grads.end())
                        acc_grads[p] = gradient;
                    else
                        acc_grads[p] = acc_grads[p] + gradient;
                }
            QueryPerformanceCounter(&endTime_bo);
            runTime_bo = (((endTime_bo.QuadPart - startTime_bo.QuadPart) * 1000.0f) / cpuFreq_bo.QuadPart);
            #ifdef TIME_BACKWARD
            cout << "basic_optimizer.hpp: backward() time: " << runTime_bo << "ms.\n";
            #endif
        }

        void step()
        {
            forward_backward();
            
            ++step_no;
            if (step_no >= batch_size)
            {
                step_no = 0;
                this->update();
                acc_grads.clear();
            }
        }

        basic_tensor<float> average_grad(grad_node<float>* p)
        {
            if (acc_grads.find(p) == acc_grads.end())
                exit(0);
            return acc_grads[p] / batch_size;
        }

        virtual void update() = 0;
    };
}
