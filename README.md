# Scalable Neural Network Kernels 
## Linearizing Deep Neural Networks via RF mechanisms :

Scalable neural network kernels (SNNKs) are novel kernel based methods which can be used to approximate regular feedforward layers (FFLs) with favorable computational properties. We
introduce the mechanism of the universal random features (or URFs) which is applied to instantiate several SNNK variants. SNNKs effectively disentangle the inputs from the parameters of the neural network in the FFL, only to connect them in the final computation via the dot-product kernel (see Figure below). They are also strictly more expressive, as allowing to model complicated relationships beyond the functions of the dot-products of parameter-input vectors. 


This repository accompanies the paper ["Scalable Neural Network Kernels"](https://openreview.net/pdf?id=4iPw1klFWa)

Arijit Sehanobish, Krzysztof Choromanski, Yunfan Zhao, Avinava Dubey, Valerii Likhosherstov

Independent Researcher, Google DeepMind & Columbia University, Harvard University, Google Research, Waymo.

The Twelfth International Conference on Learning Representations (ICLR), 2024

<p align="center">
<img src="https://github.com/arijitthegame/neural-network-kernels/blob/main/main-figure-4-1.png"  width="800px"/>
</p>

## Installation
```bash
git clone git@github.com:arijitthegame/neural-network-kernels.git
cd neural-network-kernels
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e . --user
```

## Getting Started
All code resides in the `src` folder. The `nnk` subfolder contains the implementation of our custom SNNK layers and SNNK-adapter layers. 

NLP experiments can be found in the `text_classification` subfolder. The subfolder contains the following implementations : 
- `custom_modeling_bert` : Custom BERT model with linearized tanh pooler layer
- `relu_adapter_bert` : BERT model with ReLU-SNNK Adapters
- `custom_adapter_bert` : BERT model with sine/cosine-SNNK Adapters
- `uptrain_bert` : Custom BERT model with various MLP blocks linearized by SNNK layers

Vision experiments can be found in the `vision` subfolder. The subfolder contains the following implementations : 
- `modeling_custom_vit` : Custom ViT model with linearized tanh pooler layer
- `relu_adapter_vit` : ViT model with ReLU-SNNK Adapters
- `custom_adapter_vit` : ViT model with sine/cosine-SNNK Adapters
- `uptrain_vit` : Custom ViT model with various MLP blocks linearized by SNNK layers
- `finetune_vit_cifar` : Simple script to show how to use any of the custom ViT models for image classification. Uses HF Trainer.


## Citation
If you find our work useful, please cite : 

```bibtex
@inproceedings{sehanobish2023scalable,

  title={Scalable Neural Network Kernels},
  
  author={Sehanobish, Arijit and Choromanski, Krzysztof Marcin and Zhao, Yunfan and Dubey, Kumar Avinava and Likhosherstov, Valerii},
  
  booktitle={The Twelfth International Conference on Learning Representations},
  
  year={2023}
}
```



