from functools import reduce


def recursive_set_requires_grad(module, requires_grad):
    for _key, p in module.named_parameters():
        p.requires_grad = requires_grad


def get_module_by_name(module, access_string):
    """
    Gets a module by its name in dotted notation

    :param module: The parent module to get the module from
    :param access_string: name of the module of interest
    :return: module of interest

    Source:
    https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8#accessing-nested-child-by-name-3
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
