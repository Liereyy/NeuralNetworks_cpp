#include "tensor.hpp"
#include "basic_tensor_gen.hpp"
#include "tensor_gen.hpp"
#include <fstream>
#include <ctime>
#include "tensor_operator.hpp"
#include "optimizers.hpp"
#include "basic_node.hpp"
using namespace torch;

int main()
{
    srand(time(0));
    ofstream fout("out.txt");
    auto X = randn({10, 10});
    X.gnode->train_able = false;
    auto L1 = Conv2d(X, pair<int, int>(3, 3), pair<int, int>(1, 1));
    auto P1 = MaxPool2d(L1, pair<int, int>(2, 2), pair<int, int>(2, 2));
    auto L2 = Conv2d(P1, pair<int, int>(3, 3), pair<int, int>(1, 1));
    auto P2 = MaxPool2d(L2, pair<int, int>(2, 2), pair<int, int>(2, 2));
    auto A = randn({1, 6});
    auto prob = P2 * A;
    auto labels = tensor<float>({0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {1, 6});
    labels.gnode->train_able = false;
    auto loss = CrossEntropyLoss(prob, labels);
    loss.gnode->forward();
    // fout << X << L1 << P1 << L2 << A << prob << loss;
    // fout << X.gnode->jacobi << L1.gnode->jacobi << P1.gnode->jacobi << L2.gnode->jacobi;

    GD optim(loss.gnode, 1e-2, 10);

    const int train_times = 30000;
    for (int i = 0; i < train_times; ++i)
        optim.step();
    loss.gnode->forward();
    fout << "after optimized:\n";
    fout << "prob:" << prob;
    fout << "loss:" << loss;

    return 0;
}