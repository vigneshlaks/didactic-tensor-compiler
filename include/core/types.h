#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

enum Precision {
    Int8,
    Float16,
    Float32
};

class Tensor {
private:
    void fillStride();

public:
    std::vector<float> storage = {};
    std::vector<size_t> dimension = {};
    std::vector<float> grad = {};
    std::vector<size_t> stride = {};

    float* d_storage = nullptr;
    float* d_grad = nullptr;
    bool onDevice = false;

    Tensor(std::vector<size_t> dimension);
    Tensor(std::vector<size_t> dimension, std::vector<float> storage);

    float getValue(std::vector<size_t> index);
    void setValue(std::vector<size_t> index, float value);
    float getGrad(std::vector<size_t> index);
    void setGrad(std::vector<size_t> index, float value);
    void accumulateGrad(std::vector<size_t> index, float value);
    std::string print();
    std::string printVerbose();

    void toDevice();
    void toHost();
    void freeDevice();
    void zeroGrad();
    void setGradElement(size_t index, float value);
    void sgdUpdate(float lr);
    ~Tensor();
};

#endif