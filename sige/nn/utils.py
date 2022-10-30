import torch


def activation(x: torch.Tensor, activation_name: str):
    if activation_name == "relu":
        return torch.relu(x)
    elif activation_name == "sigmoid":
        return torch.sigmoid(x)
    elif activation_name == "tanh":
        return torch.tanh(x)
    elif activation_name == "swish":
        return x * torch.sigmoid(x)
    elif activation_name == "identity":
        return x
    else:
        raise ValueError("Unknown activation: [%s]!!!" % activation_name)
