import sys

absolute_path = "/Users/arijitsehanobish/neural-network-kernels/src/nnk/"
sys.path.insert(1, absolute_path)

import torch
import torch.nn as nn

from transformers import BertLayer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder
from transformers.modeling_outputs import SequenceClassifierOutput

from custom_adapter import CustomReluAdapter


class ReluAdapterBertSelfOutput(nn.Module):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        **kwargs
    ):
        super().__init__()

        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mh_adapter = CustomReluAdapter(
            input_size=config.hidden_size,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.mh_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ReluAdapterBertOutput(nn.Module):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_adapter = CustomReluAdapter(
            input_size=config.hidden_size,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ReluAdapterBertAttention(BertAttention):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        position_embedding_type=None,
        **kwargs
    ):
        super().__init__(config)

        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.position_embedding_type = position_embedding_type
        self.constant = constant

        self.output = ReluAdapterBertSelfOutput(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )


class ReluAdapterBertLayer(BertLayer):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        self.attention = ReluAdapterBertAttention(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )
        self.output = ReluAdapterBertOutput(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )


class ReluAdapterBertEncoder(BertEncoder):
    # note this custom BERT do not support gradient checkpointing
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        self.layer = nn.ModuleList(
            [
                ReluAdapterBertLayer(
                    config=self.config,
                    num_rfs=self.num_rfs,
                    model_device=self.model_device,
                    seed=self.seed,
                    down_sample=self.down_sample,
                    init_weights=self.init_weights,
                    normalize=self.normalize,
                    normalization_constant=self.normalization_constant,
                    orthogonal=self.orthogonal,
                    constant=self.constant,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )


class ReluAdapterBertModel(BertModel):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        orthogonal,
        constant,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant

        self.encoder = ReluAdapterBertEncoder(
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )


class ReluBertForSequenceClassification(nn.Module):
    def __init__(
        self,
        config,
        num_rfs,
        model_device,
        seed,
        down_sample,
        init_weights,
        normalize,
        normalization_constant,
        constant,
        orthogonal,
        model_name_or_path,
        **kwargs
    ):
        super().__init__()
        self.num_rfs = num_rfs
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.num_labels = config.num_labels
        self.config = config
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_name_or_path = model_name_or_path

        self.bert = ReluAdapterBertModel.from_pretrained(
            self.model_name_or_path,
            config=self.config,
            num_rfs=self.num_rfs,
            model_device=self.model_device,
            seed=self.seed,
            down_sample=self.down_sample,
            init_weights=self.init_weights,
            normalize=self.normalize,
            normalization_constant=self.normalization_constant,
            orthogonal=self.orthogonal,
            constant=self.constant,
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
