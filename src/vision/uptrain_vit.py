from transformers.models.vit.modeling_vit import ViTLayer, ViTEncoder, ViTModel, ViTPreTrainedModel, ViTIntermediate, ViTForImageClassification

from nnk.nnk import NNK_Relu

class ReluNNKViTLayer(ViTLayer):
  def __init__(self, config, num_rfs, seed, normalize, normalization_constant, orthogonal, constant, model_device):
    super(ReluNNKViTLayer, self).__init__(config)
    self.config = config
    self.num_rfs = num_rfs
    self.seed = seed
    self.normalize = normalize
    self.normalization_constant = normalization_constant
    self.orthogonal = orthogonal
    self.constant = constant
    self.model_device = model_device
    ### this is a bug. Does not get initialized correctly
    #TODO: Fix this hack
    with torch.no_grad():
      self.input_weights = ViTIntermediate(config).dense.weight.data
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


class MixNNKViTEncoder(ViTEncoder) :
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
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers-self.k)]
                                )

        self.layer.extend([ReluNNKViTLayer(config, num_rfs=self.num_rfs,
                                            seed=self.seed,
                                            normalize=self.normalize,
                                            normalization_constant=self.normalization_constant,
                                            orthogonal=self.orthogonal,
                                            constant=self.constant,
                                            model_device=self.model_device
                                                     )]*self.k
                          )

class MixReluNNKViTModel(ViTModel):

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

        self.encoder = MixNNKViTEncoder(config, num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device,
                                   k = self.k)

class MixReluNNKViTForSequenceClassification(ViTForImageClassification):
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

        self.vit = MixReluNNKViTModel(config, add_pooling_layer=False,
                                       num_rfs=self.num_rfs,
                                   seed=self.seed,
                                   normalize=self.normalize,
                                   normalization_constant=self.normalization_constant,
                                   orthogonal=self.orthogonal,
                                   constant=self.constant,
                                   model_device=self.model_device,
                                   k=self.k)

