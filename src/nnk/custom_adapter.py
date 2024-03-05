import torch
from torch import nn
from nnk.nnk import NNK, NNK_Relu
from nnk.performer_attention import gaussian_orthogonal_random_matrix


class CustomAdapter(nn.Module):
    def __init__(
        self,
        input_size,
        num_rfs,
        A_fun,
        a_fun,
        xis,
        model_device,
        seed=0,
        ln_before: bool = False,
        ln_after: bool = False,
        down_sample=None,
        init_weights="mam",  # bert or mam
        normalization=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = ln_before
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.add_layer_norm_after = ln_after
        self.init_weights = init_weights
        self.orthogonal = orthogonal
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.num_rfs = num_rfs

        if self.orthogonal is False:
            self.proj_matrix = torch.rand(size=(self.num_rfs, self.input_size)).to(
                self.model_device
            )
        else:
            self.proj_matrix = gaussian_orthogonal_random_matrix(
                self.num_rfs, self.input_size, scaling=0, device=self.model_device
            )

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # print(self.init_weights)

        # if a downsample size is not passed, we just half the size of the original input

        if self.down_sample is None:
            self.initial_weights = torch.empty(self.input_size, self.input_size)
        else:
            self.initial_weights = torch.empty(self.down_sample, self.input_size)

        if self.init_weights == "bert":
            self.initial_weights = self.initial_weights.data.normal_(
                mean=0.0, std=0.02
            ).to(self.model_device)
        elif self.init_weights == "mam":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.initial_weights.data, a=math.sqrt(5))
                self.initial_weights = self.initial_weights.to(self.model_device)
        seq_list.append(
            NNK(
                self.initial_weights,
                self.A_fun,
                self.a_fun,
                self.xis,
                self.num_rfs,
                self.input_size,
                self.model_device,
                self.seed,
                self.normalize,
                self.normalization_constant,
                self.orthogonal,
                self.proj_matrix,
            )
        )
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if self.down_sample is not None:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        else:
            self.modulating_vector = torch.zeros(self.input_size)
            self.modulating_vector = nn.Parameter(self.modulating_vector)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:  # False
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if self.init_weights == "bert":
            if self.down_sample is not None:
                self.adapter_up.apply(self.init_bert_weights)

        elif self.init_weights == "mam":
            if self.down_sample is not None:
                with torch.no_grad():
                    nn.init.zeros_(self.adapter_up.weight)
                    nn.init.zeros_(self.adapter_up.bias)
        else:
            raise ValueError("Unknown init_weights type")

    def forward(self, x):
        down = self.adapter_down(x)
        if self.down_sample is not None:
            up = self.adapter_up(down)
        else:
            print(down.shape)
            print(self.modulating_vector.shape)
            up = down * self.modulating_vector

        output = x + up
        return output

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, NNK):
            module.input_weights.data.normal_(mean=0.0, std=0.02)


####################################################################################
# TODO : Too much duplicate code. To refactor


class CustomReluAdapter(nn.Module):
    def __init__(
        self,
        input_size,
        num_rfs,
        model_device,
        seed=0,
        ln_before: bool = False,
        ln_after: bool = False,
        down_sample=None,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        init_weights="mam",  # bert or mam
        constant=0.0,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = ln_before
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.add_layer_norm_after = ln_after
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.num_rfs = num_rfs

        if self.down_sample is None:
            self.initial_weights = torch.empty(self.input_size, self.input_size)
        else:
            self.initial_weights = torch.empty(self.down_sample, self.input_size)

        if self.init_weights == "bert":
            self.initial_weights = self.initial_weights.data.normal_(
                mean=0.0, std=0.02
            ).to(self.model_device)
        elif self.init_weights == "mam":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.initial_weights.data, a=math.sqrt(5))
                self.initial_weights = self.initial_weights.to(self.model_device)
        seq_list.append(
            NNK_Relu(
                input_weights=self.initial_weights,
                num_rfs=self.num_rfs,
                dim=self.input_size,
                model_device=self.model_device,
                seed=self.seed,
                normalize=self.normalize,
                normalization_constant=self.normalization_constant,
                orthogonal=self.orthogonal,
                constant=self.constant,
            )
        )
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if self.down_sample is not None:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)
        elif self.down_sample is None:
            self.modulating_vector = torch.zeros(
                self.input_size
            )  # device placement can be handled
            self.modulating_vector = nn.Parameter(self.modulating_vector)
        else:
            raise ValueError("Unsupported down sample")

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:  # False
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if self.init_weights == "bert":
            if self.down_sample is not None:
                self.adapter_up.apply(self.init_bert_weights)

        elif self.init_weights == "mam":
            if self.down_sample is not None:
                with torch.no_grad():
                    nn.init.zeros_(self.adapter_up.weight)
                    nn.init.zeros_(self.adapter_up.bias)
        else:
            raise ValueError("Unknown init_weights type")

    def forward(self, x):
        down = self.adapter_down(x)
        if self.down_sample is not None:
            up = self.adapter_up(down)
        else:
            # print(self.modulating_vector.shape)
            up = down * self.modulating_vector

        output = x + up
        return output

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, NNK):
            module.input_weights.data.normal_(mean=0.0, std=0.02)
