""""
Code for various BERT pooling experiments
"""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../nnk/"))

from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutput
import torch

import numpy as np

from nnk import input_to_rfs_torch_vectorized, create_asym_feats


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LinearBertForSequenceClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        model_name_or_path,
        A_fun: callable,
        a_fun: callable,
        xis: callable,
        num_rfs: int,
        device_model: str,
        normalization: bool,
        normalization_constant=None,
    ):
        super().__init__(
            config,
            model_name_or_path,
            A_fun,
            a_fun,
            xis,
            num_rfs,
            device_model,
            normalization,
            normalization_constant,
        )
        self.num_labels = config.num_labels
        self.config = config
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.device_model = device_model
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.model_name_or_path = model_name_or_path

        self.bert = BertModel.from_pretrained(self.model_name_or_path)
        self.w = self.bert.pooler.dense.weight.to(self.device_model)

        self.bert.pooler = Identity()

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.output_rfs = input_to_rfs_torch_vectorized(
            self.w,
            A_fun,
            a_fun,
            xis,
            num_rfs,
            self.w.shape[1],
            self.device_model,
            self.normalization,
            self.normalization_constant,
        )
        self.output_rfs = nn.Parameter(self.output_rfs)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        sequence_output = outputs[0]
        first_token_tensor = sequence_output[:, 0]
        x_rfs = input_to_rfs_torch_vectorized(
            first_token_tensor,
            self.A_fun,
            self.a_fun,
            self.xis,
            self.num_rfs,
            first_token_tensor.shape[1],
            self.device_model,
            self.normalization,
            self.normalization_constant,
        )
        pooled_output = x_rfs @ self.output_rfs.t()

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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
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


class BundledBertForSequenceClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        model_name_or_path,
        A_fun: callable,
        a_fun: callable,
        xis: callable,
        num_rfs: int,
        device_model: str,
        normalization: bool,
        normalization_constant=None,
    ):
        super().__init__(
            config,
            model_name_or_path,
            A_fun,
            a_fun,
            xis,
            num_rfs,
            device_model,
            normalization,
            normalization_constant,
        )
        self.num_labels = config.num_labels
        self.config = config
        self.A_fun = A_fun
        self.a_fun = a_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.device_model = device_model
        self.normalization = normalization
        self.normalization_constant = normalization_constant
        self.model_name_or_path = model_name_or_path

        self.bert = BertModel.from_pretrained(self.model_name_or_path)
        self.bert.pooler = Identity()

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.num_rfs, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        sequence_output = outputs[0]
        first_token_tensor = sequence_output[:, 0]
        x_rfs = input_to_rfs_torch_vectorized(
            first_token_tensor,
            self.A_fun,
            self.a_fun,
            self.xis,
            self.num_rfs,
            first_token_tensor.shape[1],
            self.device_model,
            self.normalization,
            self.normalization_constant,
        )

        pooled_output = self.dropout(x_rfs)
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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
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


class AsymKernelBertForSequenceClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        model_name_or_path,
        A_fun: callable,
        a_fun: callable,
        xis: callable,
        num_rfs: int,
        device_model: str,
        B_fun,
        b_fun,
        M,
        seed,
        normalize,
    ):
        super().__init__(
            config,
            model_name_or_path,
            A_fun,
            a_fun,
            xis,
            num_rfs,
            device_model,
            B_fun,
            b_fun,
            M,
            seed,
            normalize,
        )
        self.num_labels = config.num_labels
        self.config = config
        self.A_fun = A_fun
        self.B_fun = B_fun
        self.a_fun = a_fun
        self.b_fun = b_fun
        self.xis = xis
        self.num_rfs = num_rfs
        self.device_model = device_model
        self.M = M
        self.seed = seed
        self.normalize = normalize
        self.model_name_or_path = model_name_or_path

        self.bert = BertModel.from_pretrained(self.model_name_or_path)
        self.w = self.bert.pooler.dense.weight.to(self.device_model)
        self.b = self.bert.pooler.dense.bias.to(self.device_model)

        if self.normalize is False:
            self.w = self.w / config.hidden_size**0.25  # .5
            self.b = self.b / config.hidden_size**0.25  # .5

        self.bert.pooler = Identity()
        self.projection_matrix = torch.normal(
            mean=0, std=0.02, size=(self.num_rfs, self.config.hidden_size)
        ).to(self.device_model)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.first_rfv_plus = create_asym_feats(
            xw=self.w,
            AB_fun=self.B_fun,
            ab_fun=self.b_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=self.w.shape[1],
            device=self.device_model,
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
            xw=self.w,
            AB_fun=self.B_fun,
            ab_fun=self.b_fun,
            xis=-self.xis,  # hardcoded for a very special class of distributions. Make it more flexible
            num_rfs=self.num_rfs,
            dim=self.w.shape[1],
            device=self.device_model,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=True,
        )
        self.output_rfs = (
            (1.0 / np.sqrt(2.0))
            * np.power(1.0 + 4.0 * self.M, config.hidden_size / 4.0)
            * torch.cat([self.first_rfv_plus, self.first_rfv_minus], dim=-1)
        )

        self.output_rfs = nn.Parameter(self.output_rfs)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        sequence_output = outputs[0]
        first_token_tensor = sequence_output[:, 0]
        if self.normalize is False:
            first_token_tensor = (
                first_token_tensor / self.config.hidden_size
            )  # or normalize by \sqrt(d)
        x_rfv_plus = create_asym_feats(
            xw=first_token_tensor,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=self.xis,
            num_rfs=self.num_rfs,
            dim=first_token_tensor.shape[1],
            device=self.device_model,
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
            xw=first_token_tensor,
            AB_fun=self.A_fun,
            ab_fun=self.a_fun,
            xis=-self.xis,  # hardcoded, TODO:fix
            num_rfs=self.num_rfs,
            dim=first_token_tensor.shape[1],
            device=self.device_model,
            seed=self.seed,
            normalize=self.normalize,
            normalization_constant=None,
            orthogonal=False,
            proj_matrix=self.projection_matrix,
            bias_term=self.b,
            M=self.M,
            is_weight=False,
        )
        x_rfs = (
            (1.0 / np.sqrt(2.0))
            * np.power(1.0 + 4.0 * self.M, self.config.hidden_size / 4.0)
            * torch.cat([x_rfv_plus, x_rfv_minus], dim=-1)
        )

        pooled_output = (x_rfs @ self.output_rfs.t()).real
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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
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
