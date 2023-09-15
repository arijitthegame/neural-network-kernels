import random
import math
import numpy as np
import torch

from scipy.special import binom
from scipy.special import factorial

from scipy.stats import rv_discrete

# breaking up the power series of gelu into positive and negative

def generate_rademacher_samples(shape, complex_weights=False, device='cpu'):
    """ Draws uniformly from the (complex) Rademacher distribution. """
    if complex_weights:
        support = torch.tensor([1j, -1j, 1, -1], dtype=torch.complex64, device=device)
    else:
        support = torch.tensor([1, -1], dtype=torch.float32, device=device)
    #samples = torch.index_select(support, 0, torch.randint(len(support), shape).view(-1))
    #return samples.reshape(shape)
    indices = torch.randint(len(support), shape)
    return support[indices]


class P_Measure(rv_discrete):
    """
    The "external measure" proposed in Kar & Karnick 2012.
    """

    def __new__(cls, *args, **kwargs):
        # __new__ is called before __init__
        return rv_discrete.__new__(cls, kwargs)

    def __init__(self, p, h01, max_val, **kwargs):
        """
        p: Parameter for sampling distribution 1./p**(k+1)
        We need p>1, p=2 leads to normalized pmf.
        h01: Whether to set p(0)=p(1)=0
        """
        if p <= 1:
            raise RuntimeError('p needs to be greater than 1!')

        self.p = p
        self.h01 = h01
        self.max_val = max_val
        self.has_constant = True
        super(P_Measure, self).__init__(**kwargs)

    def _pmf(self, k):
        if self.max_val == np.inf:
            norm_const = 1./(self.p-1.)
        else:
            norm_const = np.sum(np.array(
                [1./self.p**(n+1.) for n in range(self.max_val+1)]
            ))
        pmf_vals = 1./(self.p**(k+1.))
        pmf_vals[k > self.max_val] = 0

        # there is no point in wasting features on the constant
        pmf_vals[k==0] = 0
        norm_const = norm_const - 1./self.p

        if self.h01:
            norm_const = norm_const - 1./(self.p**2)
            pmf_vals[k==1] = 0

        return pmf_vals / norm_const
    

def create_random_feats(inputs, num_rfs, degree, coef_fun_pos, coef_fun_neg,  p=2, M=1, seed=0):
  
 
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  measure = P_Measure(p, True, degree)

  feats = num_rfs - 1

  degrees = measure.rvs(size=feats)
  # print(degrees)
        # degrees are sorted from highest to lowest
  degrees, proj_dims = np.unique(np.array(degrees), return_counts=True)

  coefs_pos = [coef_fun_pos(x) for x in degrees]
  coefs_neg = [coef_fun_neg(x) for x in degrees]

  # ensures unbiasedness of maclaurin estimator
  coefs_pos /= measure._pmf(degrees)
  coefs_neg /= measure._pmf(degrees)

  modules = []
  for degree, dim in zip(degrees, proj_dims):
      # we skip the constant
      # the bias and lengthscales will already be included in the data
      # FIX
      omega = generate_rademacher_samples((dim, degree, inputs.shape[-1]))
      modules.append(omega)

  # possible rescaling required
  features_pos = []
  features_neg = []
  for i in range(len(modules)):
      if len(inputs.shape) == 2 :
        a = torch.prod(torch.einsum('ij, bkj-> bik' ,inputs, modules[i]), dim=-1)
        features_pos.append(a.t() * np.sqrt(coefs_pos[i])*p) #do we need this p
        features_neg.append(a.t() * np.sqrt(coefs_neg[i])*p)
      elif len(inputs.shape) == 3:
        a = torch.prod(torch.einsum('bij, dkj-> dbik ', inputs, modules[i]), dim=-1)
        features_pos.append(torch.permute(a, (1,2,0))* np.sqrt(coefs_pos[i]) * p )
        features_neg.append(torch.permute(a, (1,2,0))* np.sqrt(coefs_neg[i]) * p )


  features_pos = torch.cat(features_pos, dim=-1)
  features_pos = features_pos / np.sqrt(num_rfs) #should it be rfs or rfs-1
  
  features_neg = torch.cat(features_neg, dim=-1)
  features_neg = features_neg / np.sqrt(num_rfs) #should it be rfs or rfs-1

  if len(inputs.shape)== 2:
    add_features_pos = torch.Tensor([coef_fun_pos(0)]).float().sqrt().repeat(len(inputs), 1)
    all_features_pos = torch.cat([add_features_pos, features_pos], dim=1)
    add_features_neg = torch.Tensor([coef_fun_neg(0)]).float().sqrt().repeat(len(inputs), 1)
    all_features_neg = torch.cat([add_features_neg, features_neg], dim=1)

  elif len(inputs.shape) == 3:
    add_features_pos = torch.Tensor([coef_fun_pos((0))]).float().sqrt().expand((features.shape[0],features.shape[1], 1))
    all_features_pos = torch.cat([add_features_pos, features_pos], dim=-1)
    add_features_neg = torch.Tensor([coef_fun_neg((0))]).float().sqrt().expand((features.shape[0],features.shape[1], 1))
    all_features_neg = torch.cat([add_features_neg, features_neg], dim=-1)

  else :
    raise ValueError('Unsupported tensor shape')

  return torch.cat([features_pos, features_neg], dim=-1)
