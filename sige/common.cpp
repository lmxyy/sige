#include <iostream>
#include <torch/extension.h>

const std::string identityName("identity"), swishName("swish");

enum OpType {
    ADD,
    MUL
};

enum ActivationType {
    IDENTITY,
    SWISH
};


inline ActivationType getActivationType(const std::string &activationName) {
    if (activationName == identityName)
        return IDENTITY;
    else if (activationName == swishName)
        return SWISH;
    else __builtin_unreachable();
}

inline bool broadcastable(const torch::Tensor &x, const torch::Tensor &y) {
    auto xSize = x.sizes();
    auto ySize = y.sizes();
    if (xSize.size() != ySize.size())
        return false;
    for (int i = 0; i < xSize.size(); ++i)
        if (xSize[i] != ySize[i] && xSize[i] != 1 && ySize[i] != 1)
            return false;
    return true;
}