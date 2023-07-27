import math
import torch 
from torch import nn

def torch_apply_along_axis(function, x, axis: int = 0):
    """
    Torch equivalent of numpy apply along axis. This function is slow and should be avoided
    https://discuss.pytorch.org/t/apply-a-function-along-an-axis/130440
    """
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)
  
def input_to_rfs_torch(xw, AB_fun, ab_fun, xis, num_rfs, dim):

    ab_coeffs = torch_apply_along_axis(ab_fun, xis, 0)
    AB_coeffs = torch_apply_along_axis(AB_fun, xis, 0)
    gs = torch.rand(size=(num_rfs, dim))
    renorm_gs = (ab_coeffs * gs.t()).t()
    dot_products = torch.einsum('ij,j->i', renorm_gs, xw)
    squared_xw = torch.sum(xw * xw)
    correction_vector = (squared_xw / 2) * ab_coeffs * ab_coeffs
    diff_vector = dot_products - correction_vector
    return (1.0 / math.sqrt(num_rfs)) * AB_coeffs * torch.exp(diff_vector)