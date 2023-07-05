
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
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm.auto import tqdm
from utils import load_lines, write_preds, CLASS_NAMES

DOC="""
Predict labels of texts using a pre-trained classifier.
"""

def load_data(path):
    texts = load_lines(path)
    data = Dataset.from_dict({"text": texts})
    return data

def main(args):
    def tokenize(examples):
        out = tokenizer(examples["text"], padding=False, truncation=True)
        return out

    # Load data
    data = load_data(args.path_test_texts)
    nb_examples = len(data['text'])
    label_list = CLASS_NAMES
    label2id = {x:i for i,x in enumerate(label_list)}
    label2id = {i:x for i,x in enumerate(label_list)}

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.path_checkpoint)
    if model.config.problem_type == "single_label_classification":
        mode = "single"
    elif model.config.problem_type == "multi_label_classification":
        mode = "multi"
    else:
        msg = f"Unrecognized problem type '{model.config.problem_type}'"
        raise RuntimeError(msg)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.path_tokenizer)

    # Tokenize data
    print("Tokenizing...")
    data = data.map(tokenize, batched=True, num_proc=1)
    data = data.remove_columns("text")
    data.set_format("torch")
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size, collate_fn=collator)
    print(f"Nb examples: {nb_examples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Nb batches: {len(data_loader)}")

    # Use GPU If available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Run prediction
    progress = tqdm(range(len(data_loader)), desc="Batches")
    model.eval()
    all_logits = []
    for i,batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
        all_logits.append(logits.cpu().detach())
        progress.update(1)

    # Write predictions
    all_logits = torch.vstack(all_logits)
    write_preds(all_logits, label_list, args.path_preds, mode)
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_checkpoint", help="Path of directory containing binary model file and config")
    p.add_argument("path_tokenizer", help="Path of directory containing the tokenizer files")
    p.add_argument("path_test_texts", help="Path of text file containing test texts (one per line)")
    p.add_argument("path_preds", help="Path of output text file containing predicted labels")
    p.add_argument("--batch_size",
                   type=int,
                   default=32)
    args = p.parse_args()
    main(args)
