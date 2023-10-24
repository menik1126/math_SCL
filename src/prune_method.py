import torch
import torch.nn as nn
import copy
from transformers import AutoModel, AutoConfig

from src import config
import numbers
USE_CUDA = torch.cuda.is_available()

import torch.nn.utils.prune as prune

def get_weights_copy(model):
    weights_path = 'weights_temp.pt'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)

def get_module_copy(model_p, named_modules):
    for [name, module], [name_p, module_p] in zip(named_modules, model_p.named_modules()):
        print("module:{} ".format(module))
        #if name_p == name:
           #print("here..............")
        module_p = copy.deepcopy(module.detach())

    # for name, module in named_modules:
    #     print("name:{}  ".format(name))
    #     print("module:{}  ".format(module))
        
    return model_p

def CopyAndPrune_model(encoder_named_modules, predict_named_modules, generate_named_modules, merge_named_modules,
                       encoder_p, predict_p, generate_p, merge_p):
    encoder_c = get_module_copy(encoder_p, encoder_named_modules)#get_weights_copy(encoder)#torch.Tensor.copy_(encoder)#copy.deepcopy(encoder)
    predict_c = get_module_copy(predict_p, predict_named_modules)#get_weights_copy(predict)#torch.Tensor.copy_(predict)#copy.deepcopy(predict)
    generate_c= get_module_copy(generate_p, generate_named_modules) #get_weights_copy(generate)#torch.Tensor.copy_(generate)#copy.deepcopy(generate)
    merge_c =   get_module_copy(merge_p, merge_named_modules)#get_weights_copy(merge)#torch.Tensor.copy_(merge)#copy.deepcopy(merge)

    # prune.remove(encoder_c, 'weight')
    # prune.remove(predict_c, 'weight')
    # prune.remove(generate_c, 'weight')
    # prune.remove(merge_c, 'weight')

    return encoder_c.cuda(), predict_c.cuda(), generate_c.cuda(), merge_c.cuda()
    
def _compute_nparams_toprune(amount, tensor_size):
    r"""Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """
    # incorrect type already checked in _validate_pruning_amount_init
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return int(round(amount * tensor_size))  # int needed for Python 2


def _validate_pruning_amount(amount, tensor_size):
    r"""Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """
    # TODO: consider removing this check and allowing users to specify
    # a number of units to prune that is greater than the number of units
    # left to prune. In this case, the tensor will just be fully pruned.

    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            "amount={} should be smaller than the number of "
            "parameters to prune={}".format(amount, tensor_size)
        )



def _validate_pruning_amount_init(amount):
    r"""Validation helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(
            "Invalid type for amount: {}. Must be int or float." "".format(amount)
        )

    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)  # so it's a float
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            "amount={} should either be a float in the "
            "range [0, 1] or a non-negative integer"
            "".format(amount)
        )


def global_unstructured_flag(parameters, flag):
    """
        parameters: 所有的module
         setattr(module, self._tensor_name, self.apply_mask(module, self.train_flag))
         mask = getattr(module, self._tensor_name + "_mask")
         module.register_parameter(name + "_orig", orig)
    
    """
    # print("flag:{}".format(flag))
    # print("parameters:{}".format(parameters))
    for module, name in parameters:
        if hasattr(module, "flag"):
           setattr(module, "flag", torch.tensor(flag).cuda())
        else:
           module.register_buffer("flag", torch.tensor(flag).cuda())
           
def global_unstructured_test_flag(parameters, flag):
    """
        parameters: 所有的module
         setattr(module, self._tensor_name, self.apply_mask(module, self.train_flag))
         mask = getattr(module, self._tensor_name + "_mask")
         module.register_parameter(name + "_orig", orig)
    
    """
    #print("flag:{}".format(flag))
    for module, name in parameters:
        if hasattr(module, "test_flag"):
           setattr(module, "test_flag", torch.tensor(flag).cuda())
        else:
           module.register_buffer("test_flag", torch.tensor(flag).cuda())

class FooBarPruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'
    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.train_flag = True
        # self.set_flag(self.train_flag)

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            # 减去90%最小的k个值
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask
    
    
    
    

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        print("FooBarPruningMethod here ===============")
        return super(L1Unstructured, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )