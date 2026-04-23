#include "core/types.h"
#include <iostream>
#include <algorithm>

#ifdef CUDA_FOUND
#include "backend/gpu_exec.h"
#endif

#ifdef METAL_FOUND
#include "backend/metal_exec.h"
#endif

Tensor::~Tensor() {
    freeDevice();
}

Tensor::Tensor(std::vector<size_t> d) : dimension(d) {
    fillStride();

    size_t totalSize = 1;
    for (size_t dim : dimension) {
        totalSize *= dim;
    }

    storage.resize(totalSize, 0.0f);
    grad.resize(totalSize, 0.0f);
};

Tensor::Tensor(std::vector<size_t> d, std::vector<float> s) : dimension(d), storage(s) {
    fillStride();

    size_t totalSize = 1;
    for (size_t dim : dimension) {
        totalSize *= dim;
    }

    if (storage.size() != totalSize) {
        throw std::invalid_argument("Storage and Dimension Can't Be Reconciled");
    }

    grad.resize(totalSize, 0.0f);
};

// gets the flat index
float Tensor::getValue(std::vector<size_t> index) {
    size_t flatIndex = 0;
    // bounds check
    for (int i = 0; i < dimension.size(); i++) {
        if (dimension[i] - 1 < index[i]) {
            throw std::invalid_argument("Invalid Tensor Indexing");
        }
    }

    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }
    return storage[flatIndex];
};

void Tensor::setValue(std::vector<size_t> index, float value) {
    size_t flatIndex = 0;

    // bounds check
    for (int i = 0; i < dimension.size(); i++) {
        if (index[i] >= dimension[i]) {
            throw std::invalid_argument("Invalid Index Array");
        }
    }

    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }

    storage[flatIndex] = value;
};

float Tensor::getGrad(std::vector<size_t> index) {
    size_t flatIndex = 0;
    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }
    return grad[flatIndex];
};

void Tensor::setGrad(std::vector<size_t> index, float value) {
    size_t flatIndex = 0;
    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }
    grad[flatIndex] = value;
};

void Tensor::accumulateGrad(std::vector<size_t> index, float value) {
    size_t flatIndex = 0;
    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }
    grad[flatIndex] += value;
};

void Tensor::fillStride() {
    stride.resize(dimension.size());
    stride.back() = 1;
    
    // specify jump sizes based on what comes after
    for (int i = dimension.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * dimension[i + 1];
    }
};

std::string Tensor::print() {
    std::string result = "[";
    for (size_t i = 0; i < dimension.size(); i++) {
        result += std::to_string(dimension[i]);
        if (i < dimension.size() - 1) result += "×";
    }
    result += "]";
    return result;
};

std::string Tensor::printVerbose() {
    std::string result = "Tensor Information:\n";

    result += "  Dimensions: [";
    for (size_t i = 0; i < dimension.size(); i++) {
        result += std::to_string(dimension[i]);
        if (i < dimension.size() - 1) result += " × ";
    }
    result += "]\n";

    result += "  Stride: [";
    for (size_t i = 0; i < stride.size(); i++) {
        result += std::to_string(stride[i]);
        if (i < stride.size() - 1) result += ", ";
    }
    result += "]\n";

    result += "  Storage Size: " + std::to_string(storage.size()) + " elements\n";

    return result;
};

void Tensor::toDevice() {
    size_t storageBytes = storage.size() * sizeof(float);
    size_t gradBytes = grad.size() * sizeof(float);

    #ifdef CUDA_FOUND
    if (d_storage == nullptr) { gpuMalloc(&d_storage, storageBytes); }
    if (d_grad    == nullptr) { gpuMalloc(&d_grad,    gradBytes);    }
    gpuCopyToDevice(d_storage, storage.data(), storageBytes);
    gpuCopyToDevice(d_grad,    grad.data(),    gradBytes);
    onDevice = true;
    #endif

    #ifdef METAL_FOUND
    if (d_storage == nullptr) { metalMalloc(&d_storage, storageBytes); }
    if (d_grad    == nullptr) { metalMalloc(&d_grad,    gradBytes);    }
    metalCopyToDevice(d_storage, storage.data(), storageBytes);
    metalCopyToDevice(d_grad,    grad.data(),    gradBytes);
    onDevice = true;
    #endif
}

void Tensor::toHost() {
    if (!onDevice || d_storage == nullptr) return;

    size_t storageBytes = storage.size() * sizeof(float);
    size_t gradBytes = grad.size() * sizeof(float);

    #ifdef CUDA_FOUND
    gpuCopyToHost(storage.data(), d_storage, storageBytes);
    gpuCopyToHost(grad.data(),    d_grad,    gradBytes);
    #endif

    #ifdef METAL_FOUND
    metalCopyToHost(storage.data(), d_storage, storageBytes);
    metalCopyToHost(grad.data(),    d_grad,    gradBytes);
    #endif
}

void Tensor::zeroGrad() {
    if (onDevice) {
        #ifdef CUDA_FOUND
        zeroDevice(d_grad, grad.size());
        return;
        #endif

        #ifdef METAL_FOUND
        metalZeroDevice(d_grad, grad.size());
        return;
        #endif
    }
    std::fill(grad.begin(), grad.end(), 0.0f);
}

void Tensor::setGradElement(size_t index, float value) {
    if (onDevice) {
        #ifdef CUDA_FOUND
        gpuCopyToDevice(d_grad + index, &value, sizeof(float));
        return;
        #endif

        #ifdef METAL_FOUND
        metalCopyToDevice(d_grad + index, &value, sizeof(float));
        return;
        #endif
    }
    grad[index] = value;
}

void Tensor::sgdUpdate(float lr) {
    if (onDevice) {
        #ifdef CUDA_FOUND
        sgdUpdateDevice(d_storage, d_grad, lr, storage.size());
        return;
        #endif

        #ifdef METAL_FOUND
        metalSgdUpdateDevice(d_storage, d_grad, lr, storage.size());
        return;
        #endif
    }
    for (size_t i = 0; i < storage.size(); i++) {
        storage[i] -= lr * grad[i];
    }
}

void Tensor::freeDevice() {
    #ifdef CUDA_FOUND
    if (d_storage != nullptr) { gpuFree(d_storage); d_storage = nullptr; }
    if (d_grad    != nullptr) { gpuFree(d_grad);    d_grad    = nullptr; }
    onDevice = false;
    #endif

    #ifdef METAL_FOUND
    if (d_storage != nullptr) { metalFree(d_storage); d_storage = nullptr; }
    if (d_grad    != nullptr) { metalFree(d_grad);    d_grad    = nullptr; }
    onDevice = false;
    #endif
}