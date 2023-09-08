import sys

sys.path.append("/src/nnk/")

from typing import Optional, Tuple
import math
import torch
from torch import nn

from transformers import (
    BertLayer,
    BertModel,
    BertPreTrainedModel,
    BertForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder


class CustomLoraBertSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        position_embedding_type=None,
        target_k=True,
        init_weights="bert",
        **kwargs,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

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

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.initial_query_weights = torch.empty(config.hidden_size, config.hidden_size)
        self.initial_value_weights = torch.empty(config.hidden_size, config.hidden_size)
        if self.target_k:
            self.initial_key_weights = torch.empty(
                config.hidden_size, config.hidden_size
            )

        if self.init_weights == "bert":
            self.initial_query_weights = self.initial_query_weights.data.normal_(
                mean=0.0, std=0.02
            ).to(self.model_device)
            self.initial_value_weights = self.initial_value_weights.data.normal_(
                mean=0.0, std=0.02
            ).to(self.model_device)
            if self.target_k:
                self.initial_key_weights = self.initial_key_weights.data.normal_(
                    mean=0.0, std=0.02
                ).to(self.model_device)

        elif self.init_weights == "mam":
            with torch.no_grad():
                nn.init.kaiming_uniform_(
                    self.initial_query_weights.data, a=math.sqrt(5)
                )
                self.initial_query_weights = self.initial_query_weights.to(
                    self.model_device
                )
                nn.init.kaiming_uniform_(
                    self.initial_value_weights.data, a=math.sqrt(5)
                )
                self.initial_value_weights = self.initial_value_weights.to(
                    self.model_device
                )
                if self.target_k:
                    nn.init.kaiming_uniform_(
                        self.initial_key_weights.data, a=math.sqrt(5)
                    )
                    self.initial_key_weights = self.initial_key_weights.to(
                        self.model_device
                    )

        else:
            raise ValueError("Unsupported initialization type")

        self.delta_query = NNK(
            self.initial_query_weights,
            self.A_fun,
            self.a_fun,
            self.xis,
            self.num_rfs,
            config.hidden_size,
            self.model_device,
            self.seed,
            self.normalize,
            self.normalization_constant,
            self.orthogonal,
        )
        self.modulating_query = nn.Parameter(
            torch.zeros(self.all_head_size), requires_grad=True
        )
        self.delta_value = NNK(
            self.initial_value_weights,
            self.A_fun,
            self.a_fun,
            self.xis,
            self.num_rfs,
            config.hidden_size,
            self.model_device,
            self.seed,
            self.normalize,
            self.normalization_constant,
            self.orthogonal,
        )
        self.modulating_value = nn.Parameter(
            torch.zeros(self.all_head_size), requires_grad=True
        )

        if self.target_k:
            self.delta_key = NNK(
                self.initial_key_weights,
                self.A_fun,
                self.a_fun,
                self.xis,
                self.num_rfs,
                config.hidden_size,
                self.model_device,
                self.seed,
                self.normalize,
                self.normalization_constant,
                self.orthogonal,
            )
            self.modulating_key = nn.Parameter(
                torch.zeros(self.all_head_size), requires_grad=True
            )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        lora_query = self.delta_query(hidden_states) * self.modulating_query

        mixed_query_layer = mixed_query_layer + lora_query

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            lora_value = self.delta_value(hidden_states) * self.modulating_value
            value_layer = value_layer + self.transpose_for_scores(lora_value)
            if self.target_k:
                lora_key = self.delta_key(hidden_states) * self.modulating_key
                key_layer = key_layer + self.transpose_for_scores(lora_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # TODO:
        # (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

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

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CustomLoraBertAttention(BertAttention):
    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        position_embedding_type=None,
        target_k=True,
        init_weights="bert",
        **kwargs,
    ):
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
        self.position_embedding_type = position_embedding_type

        self.self = CustomLoraBertSelfAttention(
            config=self.config,
            A_fun=self.A_fun,
            a_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            position_embedding_type=self.position_embedding_type,
            target_k=self.target_k,
            init_weights=self.init_weights,
            **kwargs,
        )


class CustomLoraBertLayer(BertLayer):
    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        target_k=True,
        init_weights="bert",
        **kwargs,
    ):
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
        self.position_embedding_type = config.position_embedding_type

        self.attention = CustomLoraBertAttention(
            config=self.config,
            A_fun=self.A_fun,
            a_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            position_embedding_type=self.position_embedding_type,
            target_k=self.target_k,
            init_weights=self.init_weights,
            **kwargs,
        )


class CustomLoraBertEncoder(BertEncoder):
    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        target_k=True,
        init_weights="bert",
        **kwargs,
    ):
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

        self.layer = nn.ModuleList(
            [
                CustomLoraBertLayer(
                    config=self.config,
                    A_fun=self.A_fun,
                    a_fun=self.a_fun,
                    xis=self.xis,
                    num_rfs=self.num_rfs,
                    model_device=self.model_device,
                    seed=self.seed,
                    normalize=self.normalize,
                    normalization_constant=self.normalization_constant,
                    orthogonal=self.orthogonal,
                    target_k=self.target_k,
                    init_weights=self.init_weights,
                    **kwargs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )


class CustomLoraBertModel(BertModel):
    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        target_k=True,
        init_weights="bert",
        add_pooling_layer=True,
        **kwargs,
    ):
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

        self.encoder = CustomLoraBertEncoder(
            config=self.config,
            A_fun=self.A_fun,
            a_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            target_k=self.target_k,
            init_weights=self.init_weights,
            **kwargs,
        )


class CustomLoraBertForSequenceClassification(BertForSequenceClassification):
    # You can initialize the classifier model like
    # config= AutoConfig.from_pretrained('bert-base-uncased')
    # b = CustomLoraBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config, A_fun=A_fun, a_fun=a_fun, xis=xis, num_rfs=num_rfs, model_device='cpu', seed=0,
    #                  normalize=False, normalization_constant=None, orthogonal=False,
    #                  target_k = True, init_weights='bert',
    #                  add_pooling_layer=True,
    # )

    def __init__(
        self,
        config,
        A_fun,
        a_fun,
        xis,
        num_rfs,
        model_device,
        seed=0,
        normalize=False,
        normalization_constant=None,
        orthogonal=False,
        target_k=True,
        init_weights="bert",
        **kwargs,
    ):
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
        self.num_labels = config.num_labels

        self.bert = CustomLoraBertModel(
            config=self.config,
            A_fun=self.A_fun,
            a_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            target_k=self.target_k,
            init_weights=self.init_weights,
            **kwargs,
        )
