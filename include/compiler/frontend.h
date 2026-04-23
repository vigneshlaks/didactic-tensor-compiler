#ifndef HLO_H
#define HLO_H

#include "core/ops.h"
#include "core/types.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

enum OpType {
    Const,
    Matmul,
    Relu,
    MatmulRelu,
    MSE,
    Softmax,
    CrossEntropy
};

// IR Representation
struct Node {
    std::string id;
    OpType opType;
    bool trainable = false;
    std::shared_ptr<Tensor> output;
    std::unique_ptr<Op> operation;

    Node* prev;
    Node* next;
};

struct LinkedList {
    Node* head;
    Node* tail;
    std::unordered_map<std::string, Node*> nodeMap;

};

class Pass;

struct Metadata {
    std::vector<Pass*> passes;
};

int parseBytecode();

LinkedList parseJSON(json inputIR);

Metadata parseMetaData(json inputIR);

void printLinkedList(LinkedList ll);

#endif