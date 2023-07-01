import argparse
import os
import logging
import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModelForSequenceClassification
) 

#TODO : Adapter Tuning

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, PretrainedConfig
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

peft_types_map = {'lora' : PeftType.LORA,
                'prompt_tuning' : PeftType.PROMPT_TUNING,
                'p_tuning' : PeftType.P_TUNING,
                'prefix_tuning': PeftType.PREFIX_TUNING,
                'adalora' : PeftType.ADALORA,
                'adaptation_prompt': PeftType.ADAPTION_PROMPT
    }

wandb.init(project='peft-glue-finetuning')

parser = argparse.ArgumentParser(
    description="PEFT Finetuning on GLUE on 1 GPU"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--model_name_or_path", default="roberta-large", type=str) # can use roberta-base or other bert models
parser.add_argument("--task_name", default='mrpc', type=str)
parser.add_argument("--peft_type", default='lora', choices=peft_types_map.keys())
parser.add_argument("--device", default='cuda')
parser.add_argument("--num_epochs", default=20, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument(
        "--max_length",
        type=int,
        default=128
    )
parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument('--ckpt_criterion', type=str, default='accuracy', choices=['accuracy', 'f1'])


args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

peft_type = peft_types_map['args.peft_type']

raw_datasets = load_dataset("glue", args.task_name)

if args.task_name is not None:
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
        # some defaults et here
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)


config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='right')
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, return_dict=True)
# some stupid bug to use json file to load peft_config
with open('lora_config_glue.json', 'r') as f:
    peft_config = json.load(f)

peft_config = LORAConfig(peft_config)

model = get_peft_model(model, peft_config)

if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

 # Preprocessing the datasets
if args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and args.task_name is not None
    and not is_regression
):
        # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
        label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
    else:
        logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
elif args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif args.task_name is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

padding = "max_length" if args.pad_to_max_length else False

def preprocess_function(examples):
        # Tokenize the texts
    texts = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
                # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result


processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

if args.task_name is not None:
    metric = evaluate.load("glue", args.task_name)
else:
    metric = evaluate.load("accuracy")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)


eval_dataloader = DataLoader(
    processed_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
)
if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
    test_dataset = processed_datasets["validation_mismatched"]
else :
    test_dataloader = DataLoader(
        processed_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), # kinda hardcoded. #TODO:fix
    num_training_steps=(len(train_dataloader) * num_epochs), 
)

model.to(args.device)

epochs = 0
d = {}
for epoch in range(num_epochs):
    model.train()
    loss_epoch = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(args.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss_epoch += loss.detach().numpy()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    epoch = epoch + 1
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    peft_model_id = f"{args.model_name_or_path}_{args.peft_type}_{epoch}"
    model.save_pretrained(peft_model_id)
    wandb.log({'train_loss' : loss_epoch})
    wandb.log({'val_acc': eval_metric})
    d[epoch] = eval_metric


# load best model based on validation accuracy, this ckpt line will throw an error
res = {key: d[key][args.ckpt_criterion] for key, _ in d.items()}
ckpt = max(res, key=res.get)
peft_model_path = f"{args.model_name_or_path}_{args.peft_type}_{ckpt}"
inference_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
# Load the Lora model
inference_model = PeftModel.from_pretrained(inference_model, peft_model_id) #possible bug in this loading
inference_model.eval()

for step, batch in enumerate(tqdm(test_dataloader)):
    batch.to(args.device)
    with torch.no_grad():
        outputs = inference_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )

test_metric = metric.compute()
print(f"Final test metric for task {args.task} is {test_metric}")

