import sys
sys.path.append('/src/nnk')
from nnk import NNK

from transformers import ViTModel, ViTForImageClassification, AutoConfig, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTEncoder, ViTAttention, ViTLayer
from typing import Optional, Union, Tuple

class CustomLoraViTSelfAttention(nn.Module):
    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                target_k = True, init_weights='bert', **kwargs) -> None:
        super().__init__()

        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.initial_query_weights = torch.empty(config.hidden_size, config.hidden_size)
        self.initial_value_weights = torch.empty(config.hidden_size, config.hidden_size)
        if self.target_k :
          self.initial_key_weights = torch.empty(config.hidden_size, config.hidden_size)

        if self.init_weights == 'bert':
          self.initial_query_weights = self.initial_query_weights.data.normal_(mean=0.0, std=0.02).to(self.model_device)
          self.initial_value_weights = self.initial_value_weights.data.normal_(mean=0.0, std=0.02).to(self.model_device)
          if self.target_k : 
            self.initial_key_weights = self.initial_key_weights.data.normal_(mean=0.0, std=0.02).to(self.model_device)

        elif self.init_weights == 'mam':
          with torch.no_grad():
              nn.init.kaiming_uniform_(self.initial_query_weights.data, a=math.sqrt(5))
              self.initial_query_weights = self.initial_query_weights.to(self.model_device)
              nn.init.kaiming_uniform_(self.initial_value_weights.data, a=math.sqrt(5))
              self.initial_value_weights = self.initial_value_weights.to(self.model_device)
              if self.target_k : 
                nn.init.kaiming_uniform_(self.initial_key_weights.data, a=math.sqrt(5))
                self.initial_key_weights = self.initial_key_weights.to(self.model_device)

        else : 
          raise ValueError('Unsupported initialization type')

        self.delta_query = NNK(self.initial_query_weights, self.A_fun, self.a_fun, self.xis, self.num_rfs, 
                               config.hidden_size, self.model_device, self.seed, 
                               self.normalize, self.normalization_constant, self.orthogonal
                               )
        self.modulating_query = nn.Parameter(torch.zeros(self.all_head_size), requires_grad=True)
        self.delta_value = NNK(self.initial_value_weights, self.A_fun, self.a_fun, self.xis, self.num_rfs, 
                               config.hidden_size, self.model_device, self.seed, 
                               self.normalize, self.normalization_constant, self.orthogonal
                               )
        self.modulating_value = nn.Parameter(torch.zeros(self.all_head_size), requires_grad=True)

        if self.target_k :
          self.delta_key = NNK(self.initial_key_weights, self.A_fun, self.a_fun, self.xis, self.num_rfs, 
                               config.hidden_size, self.model_device, self.seed, 
                               self.normalize, self.normalization_constant, self.orthogonal
                               )
          self.modulating_key = nn.Parameter(torch.zeros(self.all_head_size), requires_grad=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        lora_query = self.delta_query(hidden_states) * self.modulating_query
        mixed_query_layer = mixed_query_layer + lora_query

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        lora_value = self.delta_value(hidden_states) * self.modulating_value
        value_layer = value_layer + self.transpose_for_scores(lora_value)
        if self.target_k :
          lora_key = self.delta_key(hidden_states) * self.modulating_key
          key_layer = key_layer + self.transpose_for_scores(lora_key)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class CustomLoraViTAttention(ViTAttention):
    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                 target_k = True, init_weights='bert', **kwargs) -> None:
        super().__init__(config)
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights
        self.config = config
        self.attention = CustomLoraViTSelfAttention(config=self.config, A_fun=self.A_fun, a_fun=self.a_fun,
                                                    xis=self.xis, num_rfs=self.num_rfs, model_device=self.model_device,
                                                    seed=self.seed, normalize=self.normalize, 
                                                    normalization_constant=self.normalization_constant,
                                                    orthogonal=self.orthogonal, target_k=self.target_k, 
                                                    init_weights=self.init_weights, **kwargs)


class CustomLoraViTLayer(ViTLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                 target_k = True, init_weights='bert', **kwargs) -> None:
        super().__init__(config)
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights
        self.config = config
        self.attention = CustomLoraViTAttention(config=self.config, A_fun=self.A_fun, a_fun=self.a_fun,
                                                    xis=self.xis, num_rfs=self.num_rfs, model_device=self.model_device,
                                                    seed=self.seed, normalize=self.normalize, 
                                                    normalization_constant=self.normalization_constant,
                                                    orthogonal=self.orthogonal, target_k=self.target_k, 
                                                    init_weights=self.init_weights, **kwargs)


class CustomLoraViTEncoder(ViTEncoder):
    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                 target_k = True, init_weights='bert', **kwargs) -> None:
        super().__init__(config)
        self.config = config
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights

        self.layer = nn.ModuleList([CustomLoraViTLayer(config=self.config, A_fun=self.A_fun, a_fun=self.a_fun,
                                                    xis=self.xis, num_rfs=self.num_rfs, model_device=self.model_device,
                                                    seed=self.seed, normalize=self.normalize, 
                                                    normalization_constant=self.normalization_constant,
                                                    orthogonal=self.orthogonal, target_k=self.target_k, 
                                                    init_weights=self.init_weights, **kwargs) for _ in range(config.num_hidden_layers)])

class CustomLoraViTModel(ViTModel):
    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                 target_k = True, init_weights='bert', add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs):
        super().__init__(config)
        self.config = config
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights

        self.encoder = CustomLoraViTEncoder(config=self.config, A_fun=self.A_fun, a_fun=self.a_fun,
                                                    xis=self.xis, num_rfs=self.num_rfs, model_device=self.model_device,
                                                    seed=self.seed, normalize=self.normalize, 
                                                    normalization_constant=self.normalization_constant,
                                                    orthogonal=self.orthogonal, target_k=self.target_k, 
                                                    init_weights=self.init_weights, 
                                            **kwargs)

class CustomLoraViTForImageClassification(ViTForImageClassification):
    def __init__(self, config, A_fun, a_fun, xis, num_rfs, model_device, seed=0, 
                 normalize=False, normalization_constant=None, orthogonal=False, 
                 position_embedding_type=None, target_k = True, init_weights='bert', add_pooling_layer: bool = False, **kwargs) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.target_k = target_k
        self.init_weights = init_weights
        self.config = config
        self.add_pooling_layer = add_pooling_layer

        self.vit = CustomLoraViTModel(config=self.config, A_fun=self.A_fun, a_fun=self.a_fun,
                            xis=self.xis, num_rfs=self.num_rfs, model_device=self.model_device,
                            seed=self.seed, normalize=self.normalize, 
                            normalization_constant=self.normalization_constant,
                            orthogonal=self.orthogonal, target_k=self.target_k, 
                            init_weights=self.init_weights, 
                            add_pooling_layer=self.add_pooling_layer,
                            **kwargs)
