from functools import reduce


def recursive_set_requires_grad(module, requires_grad):
    for key, p in module.named_parameters():
        p.requires_grad = requires_grad


def get_module_by_name(module, access_string):
    """
    Source: https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8#accessing-nested-child-by-name-3
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
