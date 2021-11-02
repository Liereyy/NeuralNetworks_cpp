#include "tensor.hpp"
#include "basic_tensor_gen.hpp"
#include "tensor_gen.hpp"
#include <fstream>
#include <ctime>
#include "tensor_operator.hpp"
#include "optimizers.hpp"
#include "nn.hpp"
#include "F.hpp"
#include <math.h>  
#include <windows.h>
#include <assert.h>

using namespace std;
using namespace torch;


template <typename T>
int max_index_of(vector<T>& v)
{
    int index = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[index])
            index = i;
    return index + 1;
}

// CNNNet, LeNet-5
// X -> CX1 -> PX1 -> CX2 -> PX2 -> reshape1 -> prob -> loss
//  w1,b1↗    w2,b2↗                       A↗ labels↗
class CNNNet
{
public:
    nn::Conv2d<float> conv1 = nn::Conv2d<float>(3, 6, pair<int, int>(5, 5), pair<int, int>(1, 1));  // 6*28*28
    nn::MaxPool2d<float> pool1 = nn::MaxPool2d<float>(pair<int, int>(2, 2));  // 6*14*14
    nn::Conv2d<float> conv2 = nn::Conv2d<float>(6, 16, pair<int, int>(5, 5), pair<int, int>(1, 1));  // 16*10*10
    nn::MaxPool2d<float> pool2 = nn::MaxPool2d<float>(pair<int, int>(2, 2));  // 16*5*5
    nn::Reshape<float> reshape1 = nn::Reshape<float>({1, 400});
    nn::Linear<float> fc = nn::Linear<float>(400, 10);

    CNNNet() = default;

    tensor<float> forward(tensor<float>& x)
    {
        auto x1 = pool1(conv1(x));
        auto x2 = pool2(conv2(x1));
        auto x3 = reshape1(x2);
        auto x4 = fc(x3);
        return x4;
    }

    tensor<float> operator()(tensor<float>& x)
    {
        return forward(x);
    }
};

vector<float> one_hot(int i, int len)
{
    assert(i >= 1);
    vector<float> res(len, 0);
    res[i - 1] = 1;
    return res;
}

int main()
{
    srand(time(0));
    ofstream fout("out.txt");
    ifstream Xin("data.txt");
    float x[3*32*32];
    for (int i = 0; i < 3*32*32; ++i)
        Xin >> x[i];
    
    auto X = tensor<float>(x, {3, 32, 32}, TensorOptions(false));
    auto net = CNNNet();
    auto prob = net(X);
    auto labels = tensor<float>(one_hot(7, 10), {1, 10}, TensorOptions(false));
    
    auto criterion = nn::CrossEntropyLoss<float>();

    auto loss = criterion(prob, labels);
    GD optim(loss.gnode, 1e-1, 1);

    loss.gnode->forward();

    fout << "X=" << X;
    fout << "prob:" << prob;
    fout << "labels=" << labels;
    fout << "loss=" << loss;
    fout << "actual label: " << max_index_of(labels._tensor->data) << endl;
    fout << "predict label: " << max_index_of(prob._tensor->data);

    LARGE_INTEGER cpuFreq;
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    double runTime = 0.0;
    
    fout.close();
    fout = ofstream("time.txt");
    const int train_times = 10;
    for (int i = 0; i < train_times; ++i)
    {
        if (i % 100 == 0) optim.learning_rate *= 0.1;

        QueryPerformanceFrequency(&cpuFreq);
        QueryPerformanceCounter(&startTime);
        
        optim.step();

        QueryPerformanceCounter(&endTime);
        runTime = (((endTime.QuadPart - startTime.QuadPart) * 1000.0f) / cpuFreq.QuadPart);
        fout << runTime << " ms" << endl;
    }
    loss.gnode->forward();

    cout << "train finished.";
    fout.close();
    fout = ofstream("out1.txt");
    fout << "after optimized:\n";
    fout << "X=" << X;
    fout << "prob:" << prob;
    fout << "labels=" << labels;
    fout << "loss=" << loss;
    fout << "actual label: " << max_index_of(labels._tensor->data) << endl;
    fout << "predict label: " << max_index_of(prob._tensor->data);
    return 0;
}