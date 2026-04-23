#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "compiler/frontend.h"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>

class Optimizers {
protected:
    float learningRate;
    LinkedList* list;

public:
    Optimizers(float lr, LinkedList* l) : learningRate(lr), list(l) {};

    void forward(std::vector<uint8_t> input, uint8_t output);
    void backward();
    void zeroGrad();
    void initDevice();
    void syncToHost();

    virtual void descentStep() = 0;
    virtual ~Optimizers() = default;
};

class SGD : public Optimizers {
public:
    SGD(float lr, LinkedList* l) : Optimizers(lr, l) {}
    void descentStep() override;
};


#endif
