## Simple script to run various custom ViTs on CiFar.

import argparse
import random
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTModel,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

from relu_adapter_vit import ReluAdapterViTForImageClassification
from uptrain_vit import MixReluNNKViTForSequenceClassification

parser = argparse.ArgumentParser(description="Cifar classification using ViT")
parser.add_argument(
    "--model_name_or_path", default="google/vit-base-patch16-224-in21k", type=str
)
parser.add_argument("--data", default="cifar", type=str)
parser.add_argument("--remove_unused_columns", default=False)
parser.add_argument("--evaluation_strategy", default="epoch", type=str)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--per_device_train_batch_size", default=4, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
parser.add_argument("--num_train_epochs", default=40, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--logging_steps", default=10, type=int)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--metric_for_best_model", default="accuracy", type=str)
parser.add_argument("--k", default=1, type=int, help="number of layers to replace by SNNK layers")
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--method", default='adapter', type=str, help="choose between linear_pooler, adapter and uptrain")

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

metric = load_metric("accuracy")

train_ds, test_ds = load_dataset(args.data, splits=["train", "test"])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

image_processor = ViTImageProcessor.from_pretrained(args.model_name_or_path)

image_mean = image_processor.image_mean
image_std = image_processor.image_std
size = image_processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    """ "
    Apply transformations on train set
    """
    examples["pixel_values"] = [
        _train_transforms(image.convert("RGB")) for image in examples["img"]
    ]
    return examples


def val_transforms(examples):
    """
    Apply transformations on val and test
    """
    examples["pixel_values"] = [
        _val_transforms(image.convert("RGB")) for image in examples["img"]
    ]
    return examples


# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


labels = train_ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

if args.method == 'adapter' :
    model = ReluAdapterViTForImageClassification.from_pretrained(args.model_name_or_path, 
                                                                config=config, 
                                                                num_rfs=64,
                                                                model_device=args.device,
                                                                seed=args.seed,
                                                                down_sample=None,
                                                                init_weights='mam',
                                                                ignore_mismatched_sizes = True, 
                                                                # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
                                                                )

elif args.method == 'uptrain' :
    base_model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  
            ).to(args.device)
    model = MixReluNNKViTForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                             num_rfs=8,
                                                             seed=args.seed, normalize=False,
                                                             normalization_constant=None, #config.hidden_size**.25
                                                             orthogonal=False,
                                                             constant=0, 
                                                             model_device=args.device,
                                                             k=args.k,
                                                             ignore_mismatched_sizes = True
                                                            )

    starting_wts = {}
    for i in range(12-args.k, 12):
        starting_wts[f'encoder.layer.{i}.intermediate.weights'] = phi_relu_mapping_torch(
            xw=base_pretrained_model.base_model.encoder.layer[i].intermediate.dense.weight,
            num_rand_features=8,
            dim=config.hidden_size,
            device=args.device,
            seed=args.seed,
            normalize=False,
            normalization_constant=None,
            orthogonal=False,
            constant=0.0,
            proj_matrix=None,
        )
    for i in range(12-args.k, 12):
        with torch.no_grad() :
            model.vit.encoder.layer[i].intermediate.weights.copy_(starting_wts[f'encoder.layer.{i}.intermediate.weights'])

elif args.method == 'linear_pooler' :
    num_rfs = 512
    a_fun = lambda x: torch.sin(x)
    b_fun = lambda x: torch.cos(x)
    A_fun = lambda x: torch.sin(x)
    B_fun = lambda x: torch.cos(x)

    xis_creator = lambda x: 1.0 / (2.0 * math.pi) * (x > 0.5) - 1.0 / (2.0 * math.pi) * (x < 0.5)
    if args.device == 'cuda':
        random_tosses = torch.rand(num_rfs).to(0)
    else:
        random_tosses = torch.rand(num_rfs)
    xis = xis_creator(random_tosses)
    model = LinearViTForImageClassification(config, A_fun=A_fun, 
                                            a_fun=a_fun, xis=xis, 
                                            num_rfs=num_rfs,
                                            model_device=args.device, 
                                            seed=args.seed, 
                                            normalize=False,
                                            normalization_constant=None
                                            )
else : 
    raise ValueError("Unsupported method name")

if args.method == 'adapter':
    for n, p in model.named_parameters():
        if ('adapter' in n) or ('classifier' in n) :
            p.requires_grad = (True)
        else :
            p.requires_grad = (False)
elif args.method == 'linear_pooler':
    for n, p in model.named_parameters():
        if ('output_rfs'in n) or ('classifier' in n):
            p.requires_grad = (True)
        else :
            p.requires_grad = (False)
else : 
    for n, p in model.named_parameters():
        p.requires_grad = (True)


model_name = args.model_name_or_path.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-cifar",
    remove_unused_columns=args.remove_unused_columns,
    evaluation_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    load_best_model_at_end=args.load_best_model_at_end,
    metric_for_best_model=args.metric_for_best_model,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
# TODO: Do proper logging. Right now we can just log to WandB.
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

trainer.predict(test_ds)
