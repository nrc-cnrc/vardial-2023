
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

from torch.nn import Sigmoid
from torch.nn.functional import softmax
from datasets import Dataset

CLASS_NAMES = ["BE", "CA", "CH", "FR"]

def write_preds(logits, label_list, path, mode, write_probs=False):
    assert mode in ["single", "multi"]
    if mode == "multi":
        sigmoid = Sigmoid()
        probs = sigmoid(logits).numpy()
        with open(path, 'w') as f:
            for i in range(probs.shape[0]):
                pred_classes = [(label_list[j], probs[i,j]) for  j in range(probs.shape[1]) if probs[i,j] > 0.5]
                pred_classes = sorted(pred_classes, key=lambda x:x[1], reverse=True)
                pred_strings = []
                for (c,s) in pred_classes:
                    if write_probs:
                        pred_strings.append(f"{c} {s:.5f}")
                    else:
                        pred_strings.append(f"{c}")
                f.write(" ".join(pred_strings) + "\n")
    else:
        probs = softmax(logits, dim=1)
        preds = probs.argmax(1)
        with open(path, 'w') as f:
            for doc_id, class_id in enumerate(preds):
                if write_probs:
                    out = f"{label_list[class_id]} {probs[doc_id, class_id]}"
                else:
                    out = label_list[class_id]
                f.write(out + "\n")
    return

def load_lines(path):
    with open(path) as f:
        return [line.rstrip() for line in f]

def load_labelled_data(path_texts, path_labels, mode):
    assert mode in ["single", "multi", "multi-soft"]
    texts = load_lines(path_texts)
    lines = load_lines(path_labels)
    if mode == "single":
        labels = []
        for line in lines:
            assert len(line.split(" ")) == 1
            labels.append(line)
        assert len(texts) == len(labels)
        data = Dataset.from_dict({"text": texts, "label": labels})
        return data
    labels = {c:[] for c in CLASS_NAMES}
    if mode == "multi":
        for line in lines:
            pos = set(line.split(" "))
            for c in CLASS_NAMES:
                if c in pos:
                    labels[c].append(1)
                else:
                    labels[c].append(0)
    else:
        assert mode == "multi-soft"
        for line in lines:
            pos2prob = {}
            elems = line.split(" ")
            assert len(elems) % 2 == 0
            for i in range(0,len(elems),2):
                c = elems[i]
                s = float(elems[i+1])
                pos2prob[c] = s
            for c in CLASS_NAMES:
                if c in pos:
                    labels[c].append(pos[c])
                else:
                    labels[c].append(0)
    data = labels
    data["text"] = texts
    data = Dataset.from_dict(data)
    return data
