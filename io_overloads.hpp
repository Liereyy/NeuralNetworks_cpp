#pragma once
#include <iostream>
#include <vector>
#include <array>
using namespace std;

ostream& operator<<(ostream& os, const pair<size_t, size_t>& p)
{
    os << '(' << p.first << ", " << p.second << ')';
    return os;
}

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    if (v.size() == 0) return os;
    os << '[';
    for (size_t i = 0; i < v.size() - 1; ++i)
        os << v[i] << ", ";
    os << v.back() << "]";
    return os;
}

template <typename T, size_t n>
ostream& operator<<(ostream& os, const array<T, n>& a)
{
    if (a.size() == 0) return os;
    os << '[';
    for (size_t i = 0; i < a.size() - 1; ++i)
        os << a[i] << ", ";
    os << a.back() << "]";
    return os;
}
