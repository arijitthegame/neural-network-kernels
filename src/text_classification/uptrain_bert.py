from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, BertModel, BertPreTrainedModel, BertIntermediate, BertForSequenceClassification

from nnk.nnk import NNK_Relu

class ReluNNKBertLayer(BertLayer):
  def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device):
    super(ReluNNKBertLayer, self).__init__(config)
    self.config = config
    self.num_rfs = num_rfs
    self.seed = seed
    self.normalize = normalize
    self.normalization_constant = normalization_constant
    self.orthogonal = orthogonal
    self.constant = constant
    self.model_device = model_device
    # Note this does not ionitialize with the pretrained weights. 
    # TODO : Get rid of this hack.
    with torch.no_grad():
      self.input_weights = BertIntermediate(config).dense.weight.data
      self.input_weights = self.input_weights.to(self.model_device)

    self.intermediate = NNK_Relu(self.input_weights,
                                num_rfs=self.num_rfs,
                                dim=config.hidden_size,
                                model_device=self.model_device,
                                seed=self.seed,
                                normalize=self.normalize,
                                normalization_constant=self.normalization_constant,
                                orthogonal=self.orthogonal,
                                constant=self.constant
                                 )


class ReluNNKBertEncoder(BertEncoder):
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device
        self.layer = nn.ModuleList([ReluNNKBertLayer(config, num_rfs=self.num_rfs,
                                            seed=self.seed,
                                            normalize=self.normalize,
                                            normalization_constant=self.normalization_constant,
                                            orthogonal=self.orthogonal,
                                            constant=self.constant,
                                            model_device=self.model_device
                                                     ) for _ in range(config.num_hidden_layers)]
                                   )

class MixNNKBertEncoder(BertEncoder) :
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device, k=1):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device
        self.k = k
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers-self.k)]
                                   )
        self.layer.extend([ReluNNKBertLayer(config, num_rfs=self.num_rfs,
                                            seed=self.seed,
                                            normalize=self.normalize,
                                            normalization_constant=self.normalization_constant,
                                            orthogonal=self.orthogonal,
                                            constant=self.constant,
                                            model_device=self.model_device
                                                     )]*self.k
                          )

class ReluNNKBertModel(BertModel):
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device

        self.encoder = ReluNNKBertEncoder(config, num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device)

class MixReluNNKBertModel(BertModel):
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device, add_pooling_layer=True, k=1):
        super().__init__(config)
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device
        self.k = k

        self.encoder = MixNNKBertEncoder(config, num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device,
                                   k = self.k)

class ReluNNKBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device

        self.bert = ReluNNKBertModel(config, num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device)

class MixReluNNKBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device, k=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.num_rfs = num_rfs
        self.seed = seed
        self.normalize = normalize
        self.normalization_constant = normalization_constant
        self.orthogonal = orthogonal
        self.constant = constant
        self.model_device = model_device
        self.k = k

        self.bert = MixReluNNKBertModel(config, num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device,
                                   k=self.k)
