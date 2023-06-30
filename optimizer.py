from torch.optim import *


def get_optimizer(model, optimizer, lr, weight_decay):
    optimizer = eval(optimizer)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }]

    return optimizer(params, lr=lr)
