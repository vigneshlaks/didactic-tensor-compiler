#include <iostream>
#include <algorithm>
#include "compiler/optimizers.h"
#include "core/ops.h"

// everything related to parameters
// might have to change params for other inputs
// places inputs before computation
void Optimizers::forward(std::vector<uint8_t> input, uint8_t output) {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (dynamic_cast<ConstOp*>(current->operation.get()) != nullptr && !current->trainable) {
            // fill Tensor with normalized input
            float maxVal = *std::max_element(input.begin(), input.end());
            for (int i = 0; i < current->output->storage.size(); i++) {
                current->output->storage[i] = (maxVal > 0) ? input[i] / maxVal : 0.0f;
            }
            if (current->output->onDevice) {
                current->output->toDevice();  // sync input to GPU
            }
        } else if (LossOp* lossOp = dynamic_cast<LossOp*>(current->operation.get())) {
            // one hot encode
            for (size_t i = 0; i < lossOp->groundTruth->storage.size(); i++) {
                lossOp->groundTruth->storage[i] = 0.0f;
            }
            lossOp->groundTruth->storage[output] = 1.0f;
            if (lossOp->groundTruth->onDevice) {
                lossOp->groundTruth->toDevice();  // sync ground truth to GPU
            }
        }

        // calling computation
        if (current->operation != nullptr) {
            current->operation->forward();
        }

        current = current->next;
    }
};

void Optimizers::backward() {
    Node *current = list->tail;

    // Set gradient to 1.0
    list->tail->output->setGradElement(0, 1.0f);

    while (current != nullptr) {
        if (current->operation != nullptr) {
            current->operation->backward();
        }
        current = current->prev;
    }
};

void Optimizers::zeroGrad() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->output != nullptr) {
            current->output->zeroGrad();
        }
        current = current->next;
    }
};

void Optimizers::initDevice() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (LossOp* lossOp = dynamic_cast<LossOp*>(current->operation.get())) {
            lossOp->groundTruth->toDevice();
        }

        if (current->output != nullptr) {
            current->output->toDevice();
        }
        
        current = current->next;
    }
};

void Optimizers::syncToHost() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->output != nullptr) {
            current->output->toHost();
        }
        current = current->next;
    }
};

void SGD::descentStep() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        // check if operation is trainable
        // really only matmul
        if (current->trainable && current->output != nullptr) {
            current->output->sgdUpdate(learningRate);
        }
        current = current->next;
    }
};