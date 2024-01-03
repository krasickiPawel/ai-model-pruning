import torch
import torch.nn.utils.prune as prune


def prune_worst(model, prune_amount):
    for module in get_conv_layers(model):
        prune.l1_unstructured(module, name="weight", amount=prune_amount)     # verify bias and l1_unstructued


def prune_random(model, prune_amount=0.3):
    for module in get_conv_layers(model):
        prune.random_unstructured(module, name="weight", amount=prune_amount)


def get_conv_layers(model):
    return [module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())]


# custom pruning in the future
