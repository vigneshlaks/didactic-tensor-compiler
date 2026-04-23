
#ifndef PASSES_H
#define PASSES_H

#include <vector>
#include "compiler/frontend.h"
#include "core/ops.h"
#include "core/types.h"

class Pass {
public:
    virtual ~Pass() = default;
    virtual int globalApply(LinkedList* list) = 0;
};

// fuse operations
class FusionPass : public Pass {
private:
    bool canFuse(Node *first, Node *second);
    void fuseNodes(LinkedList *list, Node *first, Node *second);
public:
    int globalApply(LinkedList* list) override;
};

// add in quantization operation(s)
class QuantizationPass : public Pass {
private:
    Precision precision;
public:
    QuantizationPass(Precision p) : precision(p) {};
    int globalApply(LinkedList* list) override;
};

// choose a backend (cpu or gpu)
class BackendPass : public Pass {
private:
    Backend backend;
public:
    BackendPass(Backend b) : backend(b) {}
    int globalApply(LinkedList* list) override;
};

// infer output shapes from input shapes
class ShapeInferencePass : public Pass {
private:
    int inferShape(Node* node);
public:
    int globalApply(LinkedList* list) override;
};

class PassManager
{
private:
    LinkedList* linkedList;
    std::vector<Pass*> passes;
public:
    PassManager(LinkedList* cg, std::vector<Pass*> p) : linkedList(cg), passes(p) {};
    void registerPass(Pass *pass);

    void runGlobal();
    bool verify();
};

#endif