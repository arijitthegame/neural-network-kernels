import torch
import math
from torch import nn

from performer_attention import gaussian_orthogonal_random_matrix

def torch_apply_along_axis(function, x, axis: int = 0):
    """
    Torch equivalent of numpy apply along axis. This function is slow and should be avoided
    https://discuss.pytorch.org/t/apply-a-function-along-an-axis/130440
    """
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=axis)
    ], dim=axis)


def input_to_rfs_torch_vectorized(xw, AB_fun, ab_fun, xis, num_rfs, dim, device, 
                                  seed=0, normalize=False, normalization_constant=None,
                                  orthogonal=False):
    if normalize :
      if normalization_constant is None :
        xw = torch.nn.functional.normalize(xw)
      else :
        xw = normalization_constant*torch.nn.functional.normalize(xw)

    ab_coeffs = torch_apply_along_axis(ab_fun, xis, 0)
    AB_coeffs = torch_apply_along_axis(AB_fun, xis, 0)
    torch.manual_seed(seed)
    if device == 'cpu':
      if orthogonal is False :
        gs = torch.rand(size=(num_rfs, dim))
      else : 
        gs = gaussian_orthogonal_random_matrix(num_rfs, dim, scaling = 0, device = 'cpu')
    else :
      if orthogonal is False :
        gs = torch.rand(size=(num_rfs, dim)).cuda()
      else :
        gs = gaussian_orthogonal_random_matrix(num_rfs, dim, scaling = 0, device = 'cuda')

    renorm_gs = (ab_coeffs * gs.t()).t()
    if len(xw.shape) == 2 :
      dot_products = torch.einsum('ij,jk->ik', xw, renorm_gs.t())
    elif len(xw.shape) == 3:
      dot_products = torch.einsum('bij,jk->bik', xw, renorm_gs.t())
    else :
      raise ValueError("Unsuported Tensor shape")
    squared_xw = torch.sum(torch.mul(xw, xw), dim=-1) #do not keepdims here
    if len(squared_xw.shape) == 1 :
      correction_vector = torch.outer(squared_xw / 2, torch.mul(ab_coeffs, ab_coeffs))
    elif len(squared_xw.shape) == 2 :
      correction_vector = torch.einsum('pq, r->pqr', squared_xw, torch.mul(ab_coeffs, ab_coeffs))
    else :
      raise ValueError("Unsupported tensor shape of xw")
    diff_vector = dot_products - correction_vector
    return (1.0 / math.sqrt(num_rfs)) * AB_coeffs * torch.exp(diff_vector)


class NNK(nn.Module) :
  def __init__(self, input_weights, A_fun, a_fun, xis, num_rfs, dim, model_device, seed=0, normalize=False, normalization_constant=None):
        super().__init__()
        self.input_weights = input_weights
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.dim = dim
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant

        self.weights = input_to_rfs_torch_vectorized(self.input_weights, self.A_fun, self.a_fun, self.xis, \
                                                     self.num_rfs, self.dim, self.model_device, self.seed,
                                                     self.normalize, self.normalize_constant)
        self.weights = nn.Parameter(self.weights)
        # TODO: ADD BIAS

  def forward(self, x):
        output_x = input_to_rfs_torch_vectorized(x, self.A_fun, self.a_fun, self.xis, self.num_rfs, self.dim, self.model_device, self.seed)
        return output_x @ self.weights.t()


