#pragma once
#include "vector"
#include <iostream>
using namespace std;

namespace torch
{
    template <typename T>
    pair<int, int> last2_of_v(const vector<T>& v)
    {
        if (v.size() < 2)
        {
            cerr << "assist_functs: matrix_size() failed.\n";
            exit(0);
        }
        return pair<int, int>(v[v.size() - 2], v.back());
    }
}