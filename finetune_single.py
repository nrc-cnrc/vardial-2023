
# Copyright (C) 2023 National Research Council Canada.
#
# This file is part of vardial-2023.
#
# vardial-2023 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# vardial-2023 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# vardial-2023. If not, see https://www.gnu.org/licenses/.

import os, argparse
from shutil import copyfile
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
from utils import load_labelled_data, write_preds, CLASS_NAMES

DOC="""
Fine-tune a pre-trained CamemBERT model to do single-label classification.
"""

MODEL_NAME = "camembert-base"

def average_weighted_averages(vals, weights):
    vals = np.array(vals)
    result = np.sum(vals * weights)/np.sum(weights)
    return result

def main(args):
    def tokenize(examples):
        out = tokenizer(examples["text"], padding=False, truncation=True)
        out["label"] = [label2id[k] for k in examples["label"]]
        return out

    # Get data
    train_data = load_labelled_data(args.path_train_texts,
                                    args.path_train_labels,
                                    "single")
    dev_data = load_labelled_data(args.path_dev_texts,
                                  args.path_dev_labels,
                                  "single")

    label_list = CLASS_NAMES
    label2id = {x:i for i,x in enumerate(label_list)}

    # Make model
    cache_dir = os.getenv("TRANSFORMERS_CACHE")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=len(label2id),
                                                               cache_dir=cache_dir)
    layer_nums = set()
    for name, param in model.named_parameters():
        if args.freeze_embeddings and name.startswith("roberta.embeddings"):
            param.requires_grad=False
        if name.startswith("roberta.encoder"):
            layer_num = int(name.split(".")[3])
            layer_nums.add(layer_num)
            if layer_num < args.freeze_encoder_upto:
                param.requires_grad=False
    nb_layers = max(layer_nums) + 1
    if args.freeze_encoder_upto > nb_layers:
        raise ValueError((f"--freeze_encoder_upto ({args.freeze_encoder_upto}) can not be "
                          f"greater than the actual number of layers ({nb_layers})"))
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"- {name} (requires_grad={param.requires_grad})")

    # Make tokenizer. Try using local cached version. This does not
    # seem to work unless the local_files_only flag is set to True.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                                  cache_dir=cache_dir,
                                                  config=model.config,
                                                  local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                                  cache_dir=cache_dir,
                                                  config=model.config)
    path_checkpoint = os.path.join(args.dir_out, "checkpoint")
    tokenizer.save_pretrained(os.path.join(path_checkpoint, "tokenizer"))

    # Tokenize data, format for pytorch native training loop. Note: I
    # only use a single CPU process because using more than one messes
    # up the logging.
    print("Tokenizing...")
    train_data = train_data.map(tokenize, batched=True, num_proc=1)
    train_data = train_data.remove_columns("text")
    train_data = train_data.rename_column("label", "labels")
    train_data.set_format("torch")
    dev_data = dev_data.map(tokenize, batched=True, num_proc=1)
    dev_data = dev_data.remove_columns("text")
    dev_data = dev_data.rename_column("label", "labels")
    dev_data.set_format("torch")
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=collator)

    # Get optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    nb_steps = args.epochs * len(train_loader)
    print(f"Batch size: {args.batch_size}")
    print(f"Nb training batches: {len(train_loader)}")
    print(f"Nb training epochs: {args.epochs}")
    print(f"Nb training steps: {nb_steps}")

    # Use GPU If available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Set up loss function
    loss_fct = CrossEntropyLoss(reduction="mean")

    # Train
    path_train_log = os.path.join(args.dir_out, "train.log")
    with open(path_train_log, 'w') as f:
        f.write("Epoch\tAvgTrainLoss\tAvgDevLoss\tDevMacroF1\n")
    best_score = -float("inf")
    path_dev_preds = os.path.join(args.dir_out, "dev_preds_latest.tsv")
    train_progress = tqdm(range(nb_steps), desc="TrainSteps")
    valid_progress = tqdm(range(len(dev_loader)), leave=False, desc="ValidSteps")
    for epoch in range(args.epochs+1):
        # Skip to validation if we just started
        if epoch > 0:
            # Train for one epoch
            model.train()
            train_losses = []
            train_batch_sizes = []
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                model_loss = outputs.loss
                weighted_loss = loss_fct(logits, batch["labels"])
                weighted_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(weighted_loss.item())
                train_batch_sizes.append(len(batch["input_ids"]))
                train_progress.update(1)
            avg_train_loss = average_weighted_averages(train_losses, train_batch_sizes)
        else:
            avg_train_loss = "N/A"

        # Save checkpoint
        model.save_pretrained(os.path.join(path_checkpoint, "latest_model"))

        # Validate
        model.eval()
        valid_progress.reset()
        all_logits = []
        all_labels = []
        valid_losses = []
        valid_batch_sizes = []
        for i,batch in enumerate(dev_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                model_loss = outputs.loss
                weighted_loss = loss_fct(logits, batch["labels"])
            all_labels.append(batch["labels"].cpu().detach())
            all_logits.append(logits.cpu().detach())
            valid_losses.append(weighted_loss.item())
            valid_batch_sizes.append(len(batch["input_ids"]))
            valid_progress.update(1)
        valid_progress.close()

        # Write predictions on dev set
        all_logits = torch.vstack(all_logits)
        write_preds(all_logits, label_list, path_dev_preds, "single")

        # Log losses and scores
        avg_valid_loss = average_weighted_averages(valid_losses, valid_batch_sizes)
        all_preds = torch.argmax(all_logits, dim=-1).numpy()
        all_labels = torch.hstack(all_labels).numpy()
        scores = f1_score(all_labels, all_preds, labels=range(len(label_list)), average=None, zero_division="warn")
        f1_ca = scores[label2id["CA"]]
        macro_f1 = np.mean(scores)
        with open(path_train_log, 'a') as f:
            f.write(f"{epoch}\t{avg_train_loss}\t{avg_valid_loss}\t{macro_f1}\n")
        if macro_f1 > best_score:
            best_score = macro_f1

            # Save checkpoint
            model.save_pretrained(os.path.join(path_checkpoint, "best_model"))

            # Save predictions on dev set
            copyfile(path_dev_preds, os.path.join(args.dir_out, "dev_preds_best.tsv"))
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_train_texts", help="Path of text file containing training texts (one per line)")
    p.add_argument("path_train_labels", help="Path of text file containing training labels (one per line)")
    p.add_argument("path_dev_texts", help="Path of text file containing dev texts (one per line)")
    p.add_argument("path_dev_labels", help="Path of text file containing dev labels (one per line)")
    p.add_argument("dir_out", help="Path of output directory")
    p.add_argument("--batch_size",
                   type=int,
                   default=8)
    p.add_argument("--epochs",
                   type=int,
                   default=3)
    p.add_argument("--freeze_embeddings",
                   action="store_true")
    p.add_argument("--freeze_encoder_upto",
                   type=int,
                   default=-1,
                   help="Freeze the lowest n layers of the encoder (one-indexed)")
    args = p.parse_args()
    if args.freeze_encoder_upto != -1 and args.freeze_encoder_upto < 1:
        raise ValueError(f"--freeze_encoder_upto ({args.freeze_encoder_upto}) must be positive")
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)
    main(args)
