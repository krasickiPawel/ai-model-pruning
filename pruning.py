import torch
import torch.nn.utils.prune as prune


def prune_worst(model, prune_amount):
    for module in get_conv_layers(model):
        prune.l1_unstructured(module, name="bias", amount=prune_amount)     # verify bias and l1_unstructued


def prune_random(model, prune_amount=0.3):
    for module in get_conv_layers(model):
        prune.random_unstructured(module, name="weight", amount=prune_amount)


def get_conv_layers(model):
    return [module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())]

#
# class RandomPruningMethod(prune.BasePruningMethod):
#     """Prune every other entry in a tensor"""
#
#     PRUNING_TYPE = "unstructured"
#
#     def compute_mask(self, t, default_mask):
#         mask = default_mask.clone()
#         mask.view(-1)[::2] = 0
#         return mask


# parameters_to_prune = [
#     (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
# ]prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.2,
# )
# def foobar_unstructured(module, name):
#     """Prunes tensor corresponding to parameter called `name` in `module`
#     by removing every other entry in the tensors.
#     Modifies module in place (and also return the modified module)
#     by:
#     1) adding a named buffer called `name+'_mask'` corresponding to the
#     binary mask applied to the parameter `name` by the pruning method.
#     The parameter `name` is replaced by its pruned version, while the
#     original (unpruned) parameter is stored in a new parameter named
#     `name+'_orig'`.
#
#     Args:
#         module (nn.Module): module containing the tensor to prune
#         name (string): parameter name within `module` on which pruning
#                 will act.
#
#     Returns:
#         module (nn.Module): modified (i.e. pruned) version of the input
#             module
#
#     Examples:
#         >>> m = nn.Linear(3, 4)
#         >>> foobar_unstructured(m, name='bias')
#     """
#     RandomPruningMethod.apply(module, name)
#     return module
