import sys
sys.path.append('/src/nnk/')

import torch 
from custom_adapter import CustomAdapter
from torch import nn

from transformers import BertLayer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertAttention, BertEncoder
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomAdapterBertSelfOutput(nn.Module):
    def __init__(self, config, num_rfs,
        A_fun,
        a_fun,
        xis,
        model_device,
        seed,
        down_sample,
        init_weights, 
        normalization=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs):
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mh_adapter = CustomAdapter(input_size=config.hidden_size, num_rfs=self.num_rfs,
        A_fun=self.A_fun,
        a_fun=self.a_fun,
        xis=self.xis,
        model_device=self.model_device,
        seed=self.seed,
        down_sample=self.down_sample,
        init_weights=self.init_weights,
        normalization=self.normalization,
        normalization_constant=self.normalization_constant,
        orthogonal=self.orthogonal,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.mh_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class CustomAdapterBertOutput(nn.Module):
    def __init__(self, config, num_rfs,
        A_fun,
        a_fun,
        xis,
        model_device,
        seed,
        down_sample,
        init_weights, 
        normalization=False,
        normalization_constant=None,
        orthogonal=False,
        **kwargs):
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_adapter = CustomAdapter(input_size=config.hidden_size, num_rfs=self.num_rfs,
        A_fun=self.A_fun,
        a_fun=self.a_fun,
        xis=self.xis,
        model_device=self.model_device,
        seed=self.seed,
        down_sample = self.down_sample,
        init_weights = self.init_weights,
        normalization=self.normalization,
        normalization_constant=self.normalization_constant,
        orthogonal=self.orthogonal,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.output_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    


class CustomAdapterBertAttention(BertAttention):
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          position_embedding_type=None,
          normalization=False,
          normalization_constant=None,
          orthogonal=False,
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.output = CustomAdapterBertSelfOutput(config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights,
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                                  )


class CustomAdapterBertLayer(BertLayer):
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          normalization=False,
          normalization_constant=None,
          orthogonal=False,
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant

        self.attention = CustomAdapterBertAttention(config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights,
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                                    )
        self.output = CustomAdapterBertOutput(config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights,
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                              )


class CustomAdapterBertEncoder(BertEncoder):
  # note this custom BERT do not support gradient checkpointing
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          normalization=False,
          normalization_constant=None,
          orthogonal=False,
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal

        self.layer = nn.ModuleList([CustomAdapterBertLayer(config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights, 
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                                  ) for _ in range(config.num_hidden_layers)])

class CustomAdapterBertModel(BertModel):
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          normalization=False,
          normalization_constant=None,
          orthogonal=False,
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
        self.normalization = normalization
        self.normalization_constant = normalization_constant

        self.encoder = CustomAdapterBertEncoder(config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights,
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                                  )


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_rfs,
          A_fun,
          a_fun,
          xis,
          model_device,
          seed,
          down_sample,
          init_weights,
          normalization=False,
          normalization_constant=None,
          model_name_or_path = 'bert-base-uncased',
          orthogonal = False,
          **kwargs):
        super().__init__()
        self.num_rfs = num_rfs
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.model_device = model_device
        self.seed = seed
        self.down_sample = down_sample
        self.init_weights = init_weights
        self.num_labels = config.num_labels
        self.config = config
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.model_name_or_path = model_name_or_path
        self.orthogonal = orthogonal


        self.bert = CustomAdapterBertModel.from_pretrained(self.model_name_or_path, config=self.config,
                                                  num_rfs=self.num_rfs,
                                                  A_fun=self.A_fun,
                                                  a_fun=self.a_fun,
                                                  xis=self.xis,
                                                  model_device=self.model_device,
                                                  seed=self.seed,
                                                  down_sample=self.down_sample,
                                                  init_weights=self.init_weights,
                                                  normalization=self.normalization,
                                                  normalization_constant=self.normalization_constant,
                                                  orthogonal=self.orthogonal,
                                                  )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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