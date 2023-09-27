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

########## Two possible ways of using it 

class mynetwork(nn.Module):
    def __init__(self, w):
        super().__init__() 
        self.w = w
        self.weights = input_to_rfs_torch(self.w, A_fun, a_fun, xis, num_rfs, dim)
        self.weights = nn.Parameter(self.weights)

    def forward(self, x):
        xb = input_to_rfs_torch(x, A_fun, a_fun, xis, num_rfs, dim)
        return xb @ self.weights 
    
###################### TEST
if __name__ == "__main__":
    dim = 5
    x = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0]).float()
    w = torch.Tensor([5.0, 4.0, 3.0, 2.0, 1.0]).float()

    bias = torch.Tensor([0.0])
    groundtruth_value = torch.cos(torch.dot(x, w)+bias)
    num_rfs = 10000
    a_fun: lambda xi: 2.0 * math.pi * 1j * xi
    b_fun: lambda x: 1
    A_fun: lambda x: torch.exp(bias)
    B_fun: lambda x: 1

    xis_creator = lambda x: 1.0 / (2.0 * math.pi) * (x > 0.5) - 1.0 / (2.0 * math.pi) * (x < 0.5)
    random_tosses = torch.rand(num_rfs)
    xis = xis_creator(random_tosses)

    x_rfs = input_to_rfs_torch(x, A_fun, a_fun, xis, num_rfs, dim)
    w_rfs = input_to_rfs_torch(w, B_fun, b_fun, xis, num_rfs, dim)

    print(torch.dot(x_rfs, w_rfs))
    print(groundtruth_value) # not great

# TEST 1
    net = mynetwork(w)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # real stupid test
    for i in range(5):
        optimizer.zero_grad()
        l = net(x)
        print(l)
        l.backward() 
        optimizer.step()


