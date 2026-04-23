#ifndef OPS_H
#define OPS_H

#include <vector>
#include <any>
#include <string>
#include <memory>
#include "core/types.h"

enum Backend
{
    CPU,
    GPU,
    METAL
};



class Op
{
public:
    Backend backend = CPU;
    int setBackend(Backend b);
    virtual ~Op() = default;
    virtual bool verify() = 0;
    virtual std::string print() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual std::vector<size_t> inferOutputShape() = 0;
    virtual void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) = 0;
};

// for loss functions / nodes / criterion
class LossOp : public Op
{
public:
    std::shared_ptr<Tensor> groundTruth;
    virtual ~LossOp() = default;
    virtual bool verify() = 0;
    virtual std::string print() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual std::vector<size_t> inferOutputShape() = 0;
    virtual void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) = 0;
};

class ConstOp : public Op
{
public:
    std::shared_ptr<Tensor> output;
    ConstOp(std::shared_ptr<Tensor> o)
        : output(o) {};

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class MatMulOp : public Op
{
public:
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> output;

    MatMulOp(std::shared_ptr<Tensor> left, std::shared_ptr<Tensor> right, std::shared_ptr<Tensor> output)
        : lhs(left), rhs(right), output(output) {};

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class ReluOp : public Op
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;

    ReluOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o)
        : input(i), output(o) {}

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class MatMulReluOp : public Op
{
public:
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> matmul_output;

    MatMulReluOp(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs, std::shared_ptr<Tensor> output)
        : lhs(lhs), rhs(rhs), output(output)
    {
        matmul_output = std::make_shared<Tensor>(output->dimension);
    }

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class QuantizationOp : public Op
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    Precision precision;
    float scale;

    QuantizationOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o)
        : input(i), output(o), scale(1.0f) {}

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class DequantizationOp : public Op
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    QuantizationOp *quantOp;

    DequantizationOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o, QuantizationOp *qOp = nullptr)
        : input(i), output(o), quantOp(qOp) {}

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class SoftmaxOp : public Op
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;

    SoftmaxOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o) :
        input(i), output(o) {};

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

// losses
class CrossEntropyOp : public LossOp
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    
    // fill parent ground truth
    CrossEntropyOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o, std::shared_ptr<Tensor> g) :
        input(i), output(o) { groundTruth = g; }

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};

class MSEOp : public LossOp
{
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;

    // fill parent ground truth
    MSEOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o, std::shared_ptr<Tensor> g)
        : input(i), output(o) { groundTruth = g; }

    bool verify() override;
    std::string print() override;
    void forward() override;
    void backward() override;
    std::vector<size_t> inferOutputShape() override;
    void updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) override;
};
#endif