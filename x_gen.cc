#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
using namespace std;

int main()
{
    ofstream fout("data.txt");
    srand(time(0));
    for (int i = 0; i < 3*32*32; ++i)
        fout << rand() * 1.0 / RAND_MAX * pow(-1, rand()) << ' ';
    return 0;
}