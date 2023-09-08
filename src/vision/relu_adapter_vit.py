import os
import sys

absolute_path = "/Users/arijitsehanobish/neural-network-kernels/src/nnk/"
sys.path.insert(1, absolute_path)

from custom_adapter import CustomReluAdapter

# add adapters to ViToutput and ViTlayer
from typing import Union, Optional, Tuple
import torch
from torch import nn
from transformers import ViTModel, ViTForImageClassification, AutoConfig
from transformers.models.vit.modeling_vit import (
    ViTEncoder,
    ViTAttention,
    ViTIntermediate,
)


class ReluAdapterViTOutput(nn.Module):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        constant=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.constant = constant
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_adapter = CustomReluAdapter(
            input_size=config.hidden_size,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            constant=self.constant,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapter(hidden_states)

        return hidden_states + input_tensor


class ReluAdapterViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        constant=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ) -> None:
        super().__init__()

        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.constant = constant
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ReluAdapterViTOutput(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            constant=self.constant,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
        )
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention_adapters = CustomReluAdapter(
            input_size=self.config.hidden_size,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            constant=self.constant,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        attention_output = self.attention_adapters(attention_output)
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class CustomAdapterViTEncoder(ViTEncoder):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        constant=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ):
        super().__init__(config)

        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.constant = constant
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.layer = nn.ModuleList(
            [
                ReluAdapterViTLayer(
                    config=self.config,
                    num_rfs=self.num_rfs,
                    model_device=self.model_device,
                    seed=self.seed,
                    down_sample=self.down_sample,
                    init_weights=self.init_weights,
                    constant=self.constant,
                    normalize=self.normalize,
                    normalization_constant=self.normalization_constant,
                    orthogonal=self.orthogonal,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )


class ReluAdapterViTModel(ViTModel):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
        constant=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ):
        super().__init__(config)

        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.add_pooling_layer = add_pooling_layer
        self.use_mask_token = use_mask_token
        self.constant = constant
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.encoder = ReluAdapterViTEncoder(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            constant=self.constant,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
        )


class ReluAdapterViTForImageClassification(ViTForImageClassification):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        add_pooling_layer: bool = False,
        use_mask_token: bool = False,
        constant=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs
    ):
        super().__init__(config)

        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.add_pooling_layer = add_pooling_layer
        self.use_mask_token = use_mask_token
        self.constant = constant
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.vit = ReluAdapterViTModel(
            config=config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            add_pooling_layer=self.add_pooling_layer,
            use_mask_token=self.use_mask_token,
            constant=self.constant,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
        )
