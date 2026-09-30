def require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch no esta instalado. Instale el extra learning del proyecto."
        ) from error
    return torch


def build_risk_mlp(input_size: int, hidden_sizes: tuple[int, ...], classes: int = 5):
    torch = require_torch()
    layers = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend((torch.nn.Linear(previous, size), torch.nn.ReLU()))
        previous = size
    layers.append(torch.nn.Linear(previous, classes))
    return torch.nn.Sequential(*layers)
