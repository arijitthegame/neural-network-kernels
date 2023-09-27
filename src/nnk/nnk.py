import torch
import math
from torch import nn

eps = 1e-10

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
    """
    Args :
        xw : Tensor of shape [b, s, dim] or [b, dim]
        AB_fun, ab_fun : Callable
        xis : vector of dimension (num_rfs)
        dim : input dimension of xw
        proj_matrix : matrix of size (num_rfs, dim)
    """
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
                gs = torch.randn(size=(num_rfs, dim))
            else:
                gs = gaussian_orthogonal_random_matrix(
                    num_rfs, dim, scaling=0, device="cpu"
                )
        else:
            if orthogonal is False:
                gs = torch.randn(size=(num_rfs, dim)).cuda()
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


def torch_apply_along_axis_asym_kernel(function, x, axis, b_term, device):
    """
    v1 version of the torch_apply_along_axis
    b_term is the bias so always should be a vector.
    """
    if b_term is None:
        batched_f = torch.vmap(function, in_dims=(0))
        return batched_f(x)
    else:
        if len(b_term.shape) == 1:
            batched_f = torch.vmap(function, in_dims=(0, 0))
            xis_broadcasted = torch.add(torch.zeros(len(b_term), 1).to(device), x)
            return batched_f(xis_broadcasted, b_term.reshape((-1, 1)))
        else:
            raise ValueError("Unsuported Tensor shape")


def create_asym_feats(
    xw,
    AB_fun,
    ab_fun,
    xis,
    num_rfs,
    dim,
    device,
    seed,
    normalize,
    normalization_constant,
    orthogonal,
    proj_matrix,
    bias_term,
    M,
    is_weight,
):
    """
    V1 version of the above and code for equation 10 in the paper,
    Note this creates asymmetric features as different features are created if the input is W or X.
    """
    if normalize:
        if normalization_constant is None:
            xw = torch.nn.functional.normalize(xw)
            bias_term = (
                bias_term - bias_term.min() / (bias_term.max() - bias_term.min()) + eps
            )
        else:
            xw = normalization_constant * torch.nn.functional.normalize(xw)
            bias_term = normalization_constant * bias_term - bias_term.min() / (
                bias_term.max() - bias_term.min() + 0.000001
            )  # this will set bias_term to be None if input bias are zeros
    ab_coeffs = torch_apply_along_axis_asym_kernel(ab_fun, xis, 0, None, device)
    AB_coeffs = torch_apply_along_axis_asym_kernel(
        AB_fun, xis, 0, bias_term, device
    )  # needs to be a vector. each entry correspond to a bias term
    # if bias is None, then we can just compute AB_coeffs once
    torch.manual_seed(seed)
    ab_coeffs = ab_coeffs.squeeze()
    ab_coeffs = ab_coeffs.to(device)
    AB_coeffs = AB_coeffs.to(device)

    if proj_matrix is None:
        if device == "cpu":
            if orthogonal is False:
                gs = torch.normal(mean=0, std=1, size=(num_rfs, dim))
            else:
                gs = gaussian_orthogonal_random_matrix(
                    num_rfs, dim, scaling=0, device="cpu"
                )
        else:
            if orthogonal is False:
                gs = torch.normal(mean=0, std=1, size=(num_rfs, dim)).to(device)
            else:
                gs = gaussian_orthogonal_random_matrix(
                    num_rfs, dim, scaling=0, device="cuda"
                )
    else:
        if device == "cpu":
            gs = proj_matrix
        else:
            gs = proj_matrix.cuda()
    renorm_gs = (np.sqrt(1 + 4.0 * M) * ab_coeffs * gs.t()).t()
    if len(xw.shape) == 2:
        # renorm_gs has complex numbers, and torch.eimsum needs both inputs to be complex doubles
        dot_products = torch.einsum(
            "ij,jk->ik", xw.type(torch.cfloat), renorm_gs.t()
        )  # complex128 / cfloat
    elif len(xw.shape) == 3:
        dot_products = torch.einsum("bij,jk->bik", xw.type(torch.cfloat), renorm_gs.t())
    else:
        raise ValueError("Unsuported Tensor shape")
    squared_xw = torch.sum(torch.mul(xw, xw), dim=-1)  # do not keepdims here
    if len(squared_xw.shape) == 1:
        correction_vector = torch.outer(squared_xw / 2, torch.mul(ab_coeffs, ab_coeffs))
    elif len(squared_xw.shape) == 2:
        correction_vector = torch.einsum(
            "pq, r->pqr", squared_xw / 2, torch.mul(ab_coeffs, ab_coeffs)
        )
    else:
        raise ValueError("Unsupported tensor shape of xw")
    correction_vector += (
        M * torch.linalg.norm(gs, axis=-1) * torch.linalg.norm(gs, axis=-1)
    )
    # check the shape of dot_products and correction vector
    diff_vector = dot_products - correction_vector
    if is_weight:
        return (1.0 / math.sqrt(num_rfs)) * AB_coeffs * torch.exp(diff_vector)
    else:  # is x
        return (1.0 / math.sqrt(num_rfs)) * torch.exp(diff_vector)


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
    """
    Args :
        xw : Tensor of shape [b, s, dim] or [b, dim]
        dim : input dimension of xw
        proj_matrix : matrix of size (num_rfs, dim)
    """
    # constant can be used to allow for some negative features if needed. (like leaky relu)
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
            gs = torch.randn(num_rand_features, dim).to(device)  # (8,32)

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
        """
        Args :
            input_weights : [b, s, dim] or [b, dim]
            A_fun, a_fun : Callable
            xis : vector of shape (num_rfs)
            dim : dimension of the input tensor
            proj_matrix : matrix of shape (num_rf, dim)

        """
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
        """ "
        Args :
            input_weights : [b, s, dim] or [b, dim]
            dim : dimension of the input tensor
            proj_matrix : matrix of shape (num_rf, dim)
        """
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
            self.projection_matrix = torch.randn(self.num_rfs, self.dim).to(
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


class NNK_Asym_Kernel(nn.Module):
    def __init__(
        self,
        input_weights,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        dim,
        model_device,
        seed,
        normalize,
        B_fun,
        b_fun,
        M,
        b,
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
        self.B_fun = B_fun
        self.b_fun = b_fun
        self.M = M
        self.b = b

        # hack using the BERT initialization here
        self.projection_matrix = torch.normal(
            mean=0, std=0.02, size=(self.num_rfs, self.dim)
        ).to(self.model_device)

        if self.normalize is False:
            self.input_weights = self.input_weights / self.dim ** 0.25  # .5
            self.b = self.b / self.dim ** 0.25  # .5

        self.first_rfv_plus = create_asym_feats(
            xw=self.input_weights,
            AB_fun=self.B_fun,
            ab_fun=self.b_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=True,
        )
        self.first_rfv_minus = create_asym_feats(
            xw=self.input_weights,
            AB_fun=self.B_fun,
            ab_fun=self.b_fun,
            xis=-self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=True,
        )
        self.weights = (
            (1.0 / np.sqrt(2.0))
            * np.power(1.0 + 4.0 * self.M, self.dim / 4.0)
            * torch.cat([self.first_rfv_plus, self.first_rfv_minus], dim=-1)
        )

        self.weights = nn.Parameter(self.weights)

    def forward(self, x):

        if self.normalize is False:
            x = x / self.dim

        x_rfv_plus = create_asym_feats(
            xw=x,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=False,
        )
        x_rfv_minus = create_asym_feats(
            xw=x,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=-self.xis,
            num_rfs=self.num_rfs,
            dim=self.dim,
            device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=False,
        )
        output_x = (
            (1.0 / np.sqrt(2.0))
            * np.power(1.0 + 4.0 * self.M, self.dim / 4.0)
            * torch.cat([x_rfv_plus, x_rfv_minus], dim=-1)
        )

        return (output_x @ self.weights.t()).real
