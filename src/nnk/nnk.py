import torch
import math
from torch import nn

from performer_attention import gaussian_orthogonal_random_matrix


def torch_apply_along_axis(function, x, axis: int = 0):
    """
    Torch equivalent of numpy apply along axis. This function is slow and should be avoided
    https://discuss.pytorch.org/t/apply-a-function-along-an-axis/130440
    """
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)


def input_to_rfs_torch_vectorized(
    xw,
    AB_fun,
    ab_fun,
    xis,
    num_rfs,
    dim,
    device,
    seed=0,
    normalize=False,
    normalization_constant=None,
    orthogonal=False,
    proj_matrix=None,
):
    if normalize:
        if normalization_constant is None:
            xw = torch.nn.functional.normalize(xw)
        else:
            xw = normalization_constant * torch.nn.functional.normalize(xw)

    ab_coeffs = torch_apply_along_axis(ab_fun, xis, 0)
    AB_coeffs = torch_apply_along_axis(AB_fun, xis, 0)
    torch.manual_seed(seed)

    if proj_matrix is None:
        if device == "cpu":
            if orthogonal is False:
                gs = torch.rand(size=(num_rfs, dim))
            else:
                gs = gaussian_orthogonal_random_matrix(
                    num_rfs, dim, scaling=0, device="cpu"
                )
        else:
            if orthogonal is False:
                gs = torch.rand(size=(num_rfs, dim)).cuda()
            else:
                gs = gaussian_orthogonal_random_matrix(
                    num_rfs, dim, scaling=0, device="cuda"
                )
    else:
        if device == "cpu":
            gs = proj_matrix
        else:
            gs = proj_matrix.cuda()
    renorm_gs = (ab_coeffs * gs.t()).t()
    if len(xw.shape) == 2:
        dot_products = torch.einsum("ij,jk->ik", xw, renorm_gs.t())
    elif len(xw.shape) == 3:
        dot_products = torch.einsum("bij,jk->bik", xw, renorm_gs.t())
    else:
        raise ValueError("Unsuported Tensor shape")
    squared_xw = torch.sum(torch.mul(xw, xw), dim=-1)  # do not keepdims here
    if len(squared_xw.shape) == 1:
        correction_vector = torch.outer(squared_xw / 2, torch.mul(ab_coeffs, ab_coeffs))
    elif len(squared_xw.shape) == 2:
        correction_vector = torch.einsum(
            "pq, r->pqr", squared_xw, torch.mul(ab_coeffs, ab_coeffs)
        )
    else:
        raise ValueError("Unsupported tensor shape of xw")
    diff_vector = dot_products - correction_vector
    return (1.0 / math.sqrt(num_rfs)) * AB_coeffs * torch.exp(diff_vector)


def phi_relu_mapping_torch(
    xw,
    num_rand_features,
    dim=None,
    seed=0,
    device="cpu",
    proj_matrix=None,
    constant=0,
    orthogonal=False,
    normalize=False,
    normalization_constant=None,
):
    # constant can be used to allow for some negative features if needed.
    if normalize:
        if normalization_constant is None:
            xw = torch.nn.functional.normalize(xw)
        else:
            xw = normalization_constant * torch.nn.functional.normalize(xw)

    if proj_matrix is None:
        torch.manual_seed(seed)
        if orthogonal:
            gs = gaussian_orthogonal_random_matrix(
                num_rand_features, dim, scaling=0, device=device
            )
        else:
            gs = torch.rand(num_rand_features, dim).to(device)  # (8,32)

    else:
        gs = proj_matrix.to(device)
    if len(xw.shape) == 2:
        dot_products = torch.einsum("ij,jk->ik", xw, gs.t())
    elif len(xw.shape) == 3:
        dot_products = torch.einsum("bij,jk->bik", xw, gs.t())
    else:
        raise ValueError("Unsuported Tensor shape")
    return (1.0 / math.sqrt(num_rand_features)) * torch.maximum(
        dot_products, constant * torch.ones_like(dot_products)
    )


class NNK(nn.Module):
    def __init__(
        self,
        input_weights,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        dim,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        proj_matrix=None,
    ):
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
        self.orthogonal = orthogonal
        self.proj_matrix = proj_matrix

        self.weights = input_to_rfs_torch_vectorized(
            xw=self.input_weights,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalize_constant=self.normalize_constant,
            orthogonal=self.orthogonal,
            proj_matrix=self.proj_matrix,
        )
        self.weights = nn.Parameter(self.weights)
        # TODO: ADD BIAS

    def forward(self, x):
        output_x = input_to_rfs_torch_vectorized(
            xw=x,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalize_constant=self.normalize_constant,
            orthogonal=self.orthogonal,
            proj_matrix=self.proj_matrix,
        )
        return output_x @ self.weights.t()


class NNK_Relu(nn.Module):
    def __init__(
        self,
        input_weights,
        num_rfs,
        dim,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        constant=0,
    ):
        super().__init__()
        self.input_weights = input_weights
        self.num_rfs = num_rfs
        self.dim = dim
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        if self.orthogonal:
            self.projection_matrix = gaussian_orthogonal_random_matrix(
                self.num_rfs, self.dim, scaling=0, device=self.model_device
            )
        else:
            self.projection_matrix = torch.rand(self.num_rfs, self.dim).to(
                self.model_device
            )

        self.weights = phi_relu_mapping_torch(
            xw=self.input_weights,
            num_rand_features=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
            proj_matrix=self.projection_matrix,
        )
        self.weights = nn.Parameter(self.weights)

    def forward(self, x):
        output_x = phi_relu_mapping_torch(
            xw=x,
            num_rand_features=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
            proj_matrix=self.projection_matrix,
        )
        return output_x @ self.weights.t()
