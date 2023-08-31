import sys
sys.path.append('/src/nnk/')
from custom_adapter import CustomAdapter

# add adapters to ViToutput and ViTlayer
from typing import Union, Optional, Tuple
import torch
from torch import nn
from transformers import ViTModel, ViTForImageClassification, AutoConfig
from transformers.models.vit.modeling_vit import ViTEncoder, ViTAttention, ViTIntermediate

class CustomAdapterViTOutput(nn.Module):
    def __init__(self, config, num_rfs,
        A_fun,
        a_fun,
        xis,
        model_device,
        seed,
        down_sample,
        init_weights, 
        **kwargs
        ) -> None:
        super().__init__()
        self.config = config
        self.num_rfs = num_rfs
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print('inside vit', config.hidden_size)
        self.output_adapter = CustomAdapter(input_size=config.hidden_size, num_rfs=self.num_rfs,
        A_fun=self.A_fun,
        a_fun=self.a_fun,
        xis=self.xis,
        model_device=self.model_device,
        seed=self.seed,
        down_sample = self.down_sample,
        init_weights = self.init_weights
        )


    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapter(hidden_states)

        return hidden_states + input_tensor

class CustomAdapterViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, 
                 config, 
                 num_rfs,
                A_fun,
                a_fun,
                xis,
                model_device,
                seed,
                down_sample,
                init_weights,
                **kwargs) -> None:
        super().__init__()

        self.config = config
        self.num_rfs = num_rfs
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = CustomAdapterViTOutput(self.config, self.num_rfs,
                                             self.A_fun, self.a_fun, self.xis,
                                             self.model_device, self.seed,
                                             self.down_sample, self.init_weights
                                             )
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_adapters = CustomAdapter(input_size=self.config.hidden_size, num_rfs=self.num_rfs,
                                                A_fun=self.A_fun,
                                                a_fun=self.a_fun,
                                                xis=self.xis,
                                                model_device=self.model_device,
                                                seed=self.seed,
                                                down_sample = self.down_sample,
                                                init_weights = self.init_weights
                                                )


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

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

    def __init__(self, config, num_rfs,
        A_fun,
        a_fun,
        xis,
        model_device,
        seed,
        down_sample,
        init_weights,
        **kwargs):
      super().__init__(config)

      self.config = config
      self.num_rfs = num_rfs
      self.A_fun = A_fun
      self.a_fun = a_fun
      self.xis = xis
      self.model_device = model_device
      self.seed = seed
      self.down_sample = down_sample
      self.init_weights = init_weights

      self.layer = nn.ModuleList([CustomAdapterViTLayer(config=self.config, num_rfs=self.num_rfs,
                                                A_fun=self.A_fun,
                                                a_fun=self.a_fun,
                                                xis=self.xis,
                                                model_device=self.model_device,
                                                seed=self.seed,
                                                down_sample = self.down_sample,
                                                init_weights = self.init_weights) for _ in range(config.num_hidden_layers)])

class CustomAdapterViTModel(ViTModel):
    # initialize the model like this :
    # m1 = CustomAdapterViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=config, num_rfs=num_rfs,
                # A_fun=A_fun,
                # a_fun=a_fun,
                # xis=xis,
                # model_device='cpu',
                # seed=0,
                # down_sample=48,
                # init_weights='mam'
                # )

    def __init__(self, config, num_rfs,
                A_fun,
                a_fun,
                xis,
                model_device,
                seed,
                down_sample,
                init_weights,
                add_pooling_layer: bool = True,
                use_mask_token: bool = False,
                **kwargs):
        super().__init__(config)

        self.config = config
        self.num_rfs = num_rfs
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.add_pooling_layer = add_pooling_layer
        self.use_mask_token = use_mask_token

        self.encoder = CustomAdapterViTEncoder(config=self.config, num_rfs=self.num_rfs,
                                                A_fun=self.A_fun,
                                                a_fun=self.a_fun,
                                                xis=self.xis,
                                                model_device=self.model_device,
                                                seed=self.seed,
                                                down_sample = self.down_sample,
                                                init_weights = self.init_weights)

class CustomAdapterViTForImageClassification(ViTForImageClassification):
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          add_pooling_layer: bool = False,
          use_mask_token: bool = False
          **kwargs,
          ) :
        super().__init__(config)

        self.num_rfs = num_rfs
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.add_pooling_layer = add_pooling_layer
        self.use_mask_token = use_mask_token

        self.vit = CustomAdapterViTModel(config=config,
                            num_rfs=self.num_rfs,
                            A_fun=self.A_fun,
                            a_fun=self.a_fun,
                            xis=self.xis,
                            model_device=self.model_device,
                            seed=self.seed,
                            down_sample = self.down_sample,
                            init_weights = self.init_weights,
                            add_pooling_layer=self.add_pooling_layer,
                            use_mask_token=self.use_mask_token)
