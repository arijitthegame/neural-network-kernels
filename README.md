# Scalable Neural Network Kernels 
Linearizing Deep Neural Networks via RF mechanisms :


This repository accompanies the paper ["Scalable Neural Network Kernels"](https://openreview.net/pdf?id=4iPw1klFWa)

Arijit Sehanobish, Krzysztof Choromanski, Yunfan Zhao, Avinava Dubey, Valerii Likhosherstov

Independent Researcher, Google DeepMind & Columbia University, Harvard University, Google Research, Waymo.

The Twelfth International Conference on Learning Representations (ICLR), 2024

<p align="center">
<img src="https://github.com/arijitthegame/neural-network-kernels/blob/main/main-figure-4-1.png"  width="800px"/>
</p>

## Installation


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


