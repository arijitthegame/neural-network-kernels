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
    AutoModelForImageClassification,
    TrainingArguments, 
    Trainer
)

parser = argparse.ArgumentParser(
    description="Cifar classification using ViT"
)
parser.add_argument("--model_name_or_path", default="google/vit-base-patch16-224-in21k", type=str)
parser.add_argument("--data", default="cifar", type=str)
parser.add_argument("--remove_unused_columns", default=False)
parser.add_argument("--evaluation_strategy", default="epoch", type=str)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--per_device_train_batch_size", default=4, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
parser.add_argument("--num_train_epochs", default=3, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--logging_steps", default=10, type=int)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--metric_for_best_model", default="accuracy", type=str)
parser.add_argument("--full_fine", default=False, type=bool)

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

metric = load_metric("accuracy")

train_ds, test_ds = load_dataset(args.data, splits=['train', 'test'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

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
    """"
    Apply transformations on train set
    """
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    """
    Apply transformations on val and test
    """
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
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

model = AutoModelForImageClassification.from_pretrained(
    args.model_name_or_path, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# finetune the pooler or the last layer 
if args.full_finetune is False :
    for n, p in model.named_parameters():
        if ('pooler'in n) or ('classifier' in n):
            p.requires_grad = True
        else :
            p.requires_grad = False
else : 
    for p in model.parameters():
        p.requires_grad = True

model_name = args.model_name_or_path.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-cifar",
    remove_unused_columns=args.remove_unused_columns,
    evaluation_strategy = args.evaluation_strategy,
    save_strategy = args.save_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    load_best_model_at_end=args.load_best_model_at_end,
    metric_for_best_model=args.metric_for_best_model
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
#TODO: Do proper logging. Right now we can just log to WandB.
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

trainer.predict(test_ds)