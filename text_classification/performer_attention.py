import math
import torch

# for pytorch head is 2nd dim or in other tensors would of shape [B,H, L, D]
def expplus_pytorch(data_orig,
            other_data, #other data should also have [B,H,F,D]
            is_query,
            projection_matrix=None,
            numerical_stabilizer=0.000001,
            normalize_data=True,
            numerical_renormalizer=True,
            extra_renormalize_exp_fun=False):

  data = data_orig
  if projection_matrix is None:
    return data_orig

  if normalize_data:
    data_normalizer = 1.0 / (data.shape[-1]**.25)
  else:
    data_normalizer = 1.0
    lengths = torch.linalg.norm(data, dim=-1, keepdim=True)
    data /= lengths

  ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
  data_dash = torch.einsum("blhd,md->blhm", data_normalizer * data,
                        projection_matrix)
  
  diag_data = data**2
  diag_data = torch.sum(diag_data, dim=-1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = torch.unsqueeze(diag_data, dim = - 1)

  # Calculating coefficients A, B of the FAVOR++ mechanism: 
  # make sure to switch all dim =1 in the original code to dim=2
  _, _, l, _ = data_orig.shape

  first_sum_of_squares = data**2
  first_sum_of_squares = torch.sum(
      first_sum_of_squares, axis=(-2, -1), keepdim=True)
  first_sum_of_squares *= (data_normalizer * data_normalizer)
  first_sum_of_squares /= l  # data.shape[1]
  second_sum_of_squares = other_data**2
  second_sum_of_squares = torch.sum(
      second_sum_of_squares, (-2, -1), keepdim=True)
  second_sum_of_squares *= (data_normalizer * data_normalizer)
  second_sum_of_squares /= l  #  other_data.shape[1]
  data_sum = torch.sum(data, dim=2, keepdim=True)
  other_data_sum = torch.sum(other_data, dim=2, keepdim=True)
  d_prod = torch.einsum("bhld,bhld->bhl", data_sum, other_data_sum)

  d_prod = torch.unsqueeze(d_prod, -1)
  d_prod *= (data_normalizer * data_normalizer)
  d_prod *= (2.0 / (l * l))
  ave = first_sum_of_squares + second_sum_of_squares + d_prod
  dim = projection_matrix.shape[-1]
  a_coeff = (1.0 / (4.0 * ave)) * (
      torch.sqrt((2.0 * ave + dim) *
                   (2.0 * ave + dim) + 8.0 * dim * ave) - 2.0 * ave - dim)
  a_coeff = (1.0 - 1.0 / a_coeff) / 8.0
  b_coeff = torch.sqrt(1.0 - 4.0 * a_coeff)
  d_coeff = (1.0 - 4.0 * a_coeff)** (dim / 4.0)
  with torch.no_grad(): #might be an overkill, will only find out the hard way when it crashes
    a_coeff.requires_grad_(False)
    b_coeff.requires_grad_(False)
    d_coeff.requires_grad_(False)

  # Calculating diag_omega for the FAVOR++ mechanism:
  diag_omega = projection_matrix**2
  diag_omega = torch.sum(
      diag_omega, dim = - 1) #3d tensor
  diag_omega = torch.unsqueeze(diag_omega, dim=0)
  diag_omega = torch.unsqueeze(diag_omega, dim=0)
  diag_omega = torch.unsqueeze(diag_omega, dim=0)
  diag_omega = a_coeff * diag_omega

  if numerical_renormalizer:
    if is_query:
      last_dims_t = len(data_dash.shape) - 1
      stab = b_coeff * torch.max(
          data_dash, dim=last_dims_t, keepdim=True)[0]
    else:
      stab = b_coeff * torch.max(data_dash)
    if extra_renormalize_exp_fun:
      extra_stab = torch.max(diag_data, dim=1, keepdim=True)[0]
      stab = torch.maximum(stab, extra_stab)

    data_dash = ratio * d_coeff * (
        torch.exp(b_coeff * data_dash - stab - diag_data + diag_omega) +
        numerical_stabilizer)
  else:
    data_dash = ratio * d_coeff * (
        torch.exp(b_coeff * data_dash - diag_data + diag_omega) +
        numerical_stabilizer)

  return data_dash
