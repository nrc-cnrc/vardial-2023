
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

import os, argparse, random, logging, pickle
from collections import Counter
from copy import copy, deepcopy
import numpy as np
from scipy.sparse import tril, find
from tqdm import tqdm
from utils import load_lines

DOC="""
Create datasets for single-label and multi-label classification, with random split.
"""

MAX_NE_PROP = 0.5

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def write_lines(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    return

def write_data(texts, text_to_labels, class_names, to_dir, part_name, mode):
    assert part_name in ["train", "dev", "test"]
    assert mode in ["single", "single-no-in-class-dups", "multi", "multi-soft"]
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    pairs = []
    for text in texts:
        if mode == "single":
            # Write as single-label with duplicates both within and across classes
            for class_ix, freq in enumerate(text_to_labels[text]):
                if freq > 0:
                    pairs += [(text, class_names[class_ix])] * freq
        elif mode == "single-no-in-class-dups":
            # Write as single-label with duplicates across classes only
            for class_ix, freq in enumerate(text_to_labels[text]):
                if freq > 0:
                    pairs.append((text, class_names[class_ix]))
        elif mode == "multi":
            # Write as multi-label with binary labels
            y = [class_names[i] for i in np.where(text_to_labels[text] > 0)[0]]
            pairs.append((text, " ".join(y)))
        elif mode == "multi-soft":
            # Write as multi-label with real labels based on relative class frequency
            N = text_to_labels[text].sum()
            label_fd = text_to_labels[text] / N
            y = []
            for i, freq in enumerate(label_fd):
                if freq > 0:
                    y.append(f"{class_names[i]} {freq:.5f}")
            pairs.append((text, " ".join(y)))
    random.shuffle(pairs)
    texts, labels = list(zip(*pairs))
    pathx = os.path.join(to_dir, f"{part_name}.txt")
    pathy = os.path.join(to_dir, f"{part_name}.labels")
    write_lines(pathx, texts)
    write_lines(pathy, labels)
    logger.info(f"  Wrote {pathx} and corresponding labels...")
    return

def main(args):
    # Check args
    assert not os.path.exists(args.dir_output)
    os.makedirs(args.dir_output)
    assert args.train_prop >= 0
    assert args.dev_prop >=0
    assert (args.train_prop + args.dev_prop) < 1
    if args.seed:
        random.seed(args.seed)

    # Load texts, labels, and similarit matrix
    logger.info(f"Loading texts from {args.path_texts} and labels from {args.path_labels}...")
    texts = load_lines(args.path_texts)
    labels = load_lines(args.path_labels)
    classes = sorted(set(labels))
    nb_classes = len(classes)
    assert len(texts) == len(labels)
    logger.info(f"Nb texts: {len(set(texts))}")
    nb_uniq_texts = len(set(texts))
    logger.info(f"Nb unique texts: {nb_uniq_texts}")
    logger.info(f"Loading similarity matrix from {args.path_pickle}...")
    with open(args.path_pickle, 'rb') as f:
        data = pickle.load(f)
        mat = data["matrix"]
        mat_labels = data["labels"]
    logger.info(f"Type of matrix: {type(mat)}")
    logger.info(f"Shape of matrix: {mat.shape}")
    logger.info(f"Nb nnz: {mat.nnz}")

    # Map texts to labels
    text_to_labels = {text:np.zeros(nb_classes, dtype=int) for text in mat_labels}
    class2id = {x:i for i,x in enumerate(classes)}
    for text, label in zip(texts, labels):
        text_to_labels[text][class2id[label]] += 1

    # Combine the labels of duplicates and near duplicates that have
    # more than one unique label (but keep the original label distribution too)
    logger.info(f"Identifying near duplicates with sim>={args.min_sim}...")
    mat = tril(mat, k=1, format='csr')
    rows, cols, vals = find(mat)
    nd = [(rows[i],cols[i]) for i in np.where(vals>=args.min_sim)[0]]
    logger.info("Identifying near duplicates that have different sets of unique labels...")
    ambig = set()
    for (i,j) in tqdm(nd):
        yi = copy(text_to_labels[mat_labels[i]])
        yj = copy(text_to_labels[mat_labels[j]])
        if any((yi > 0) ^ (yj > 0)):
            ambig.add((i,j))
    logger.info(f"Nb near duplicate pairs: {len(nd)}")
    logger.info(f"Nb ambiguous duplicate pairs: {len(ambig)}/{len(nd)}")
    logger.info(f"Combining labels of ambiguous near duplicates...")
    text_to_labels_combined = deepcopy(text_to_labels)
    for (i,j) in ambig:
        text_to_labels_combined[mat_labels[i]] += text_to_labels[mat_labels[j]]
        text_to_labels_combined[mat_labels[j]] += text_to_labels[mat_labels[i]]

    # Report some stats on the near-duplicate pairs
    text_to_nbrs = {}
    text_to_new_labels = {}
    for (i,j) in ambig:
        if i not in text_to_nbrs:
            text_to_nbrs[i] = []
            text_to_new_labels[i] = set()
        if j not in text_to_nbrs:
            text_to_nbrs[j] = []
            text_to_new_labels[j] = set()
        assert not j in text_to_nbrs[i]
        text_to_nbrs[i].append(j)
        assert not i in text_to_nbrs[j]
        text_to_nbrs[j].append(i)
        label_set_i = text_to_labels[mat_labels[i]].nonzero()[0]
        label_set_j = text_to_labels[mat_labels[j]].nonzero()[0]
        for y in label_set_i:
            if y not in label_set_j:
                text_to_new_labels[j].add(y)
        for y in label_set_j:
            if y not in label_set_i:
                text_to_new_labels[i].add(y)
    assert len(text_to_nbrs) == len(text_to_new_labels)
    logger.info(f"Stats on the {len(ambig)} near-duplicate pairs:")
    logger.info(f" - Nb unique texts: {len(text_to_nbrs)}")
    logger.info("  - Distribution of # neighbours per text:")
    fd = Counter(len(x) for x in text_to_nbrs.values())
    for val, count in sorted(fd.items()):
        pct = 100 * count / len(text_to_nbrs)
        logger.info(f"    - {val}: {count} ({pct:.2f}%)")
    logger.info("  - Distribution of # new, unique labels received from neighbours")
    fd = Counter(len(x) for x in text_to_new_labels.values())
    for val, count in sorted(fd.items()):
        pct = 100 * count / len(text_to_nbrs)
        logger.info(f"    - {val}: {count} ({pct:.2f}%)")

    # Apply filters to texts
    texts = mat_labels[:]
    assert len(texts) == nb_uniq_texts
    if args.no_ne:
        logger.info(f"Discarding texts where the proportion of $NE$ tokens is greater than {MAX_NE_PROP}...")
        to_remove = []
        for i,t in enumerate(texts):
            tokens = t.split(" ")
            token_fd = Counter(tokens)
            if "$NE$" in token_fd and (token_fd["$NE$"] / len(tokens)) > MAX_NE_PROP:
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            _ = texts.pop(i)
        nb_removed = len(to_remove)
        logger.info(f"Nb texts discarded: {nb_removed}")
    if args.nb_singles:
        singles = [i for i,t in enumerate(texts) if (text_to_labels[t] > 0).sum() > 1]
        logger.info(f"Sampling {args.nb_singles} of the {len(singles)} single-label texts...")
        random.shuffle(singles)
        to_remove = singles[args.nb_singles:]
        for i in sorted(to_remove, reverse=True):
            _ = texts.pop(i)
        nb_removed = len(to_remove)
        logger.info(f"Nb texts discarded: {nb_removed}")

    # Show distribution of number of labels per text
    label_count_fd = Counter((text_to_labels[text] != 0).sum() for text in texts)
    logger.info("Distribution of number of labels per text using ORIGINAL labels:")
    for k in sorted(label_count_fd.keys()):
        logger.info(f"  - {k}: {label_count_fd[k]}")
    label_count_fd = Counter((text_to_labels_combined[text] != 0).sum() for text in texts)
    logger.info("Distribution of number of labels per text using COMBINED labels:")
    for k in sorted(label_count_fd.keys()):
        logger.info(f"  - {k}: {label_count_fd[k]}")

    # Make random split of texts
    random.shuffle(texts)
    train_size = int(round(args.train_prop * len(texts)))
    dev_size = int(round(args.dev_prop * len(texts)))
    test_size = len(texts) - train_size - dev_size
    logger.info(f"Nb texts in training set (with train_prop={args.train_prop}): {train_size}")
    logger.info(f"Nb texts in dev set (with dev_prop={args.dev_prop}): {dev_size}")
    logger.info(f"Nb texts in test set: {test_size}")
    train_texts = texts[:train_size]
    dev_texts = texts[train_size:(train_size+dev_size)]
    test_texts = texts[-test_size:]
    logger.info("Shuffling...")
    random.shuffle(train_texts)
    random.shuffle(dev_texts)
    random.shuffle(test_texts)

    # Write various single-label and multi-label datasets based on
    # this random split of texts
    for part_name, part_texts in [("train", train_texts),
                                  ("dev", dev_texts),
                                  ("test", test_texts)]:
        # Split texts for this part into ambiguous and unambiguous
        # (according to either the original or the combined labels)
        ambig_o = []
        ambig_c = []
        unambig_o = []
        unambig_c = []
        for text in part_texts:
            if (text_to_labels[text] > 0).sum() > 1:
                ambig_o.append(text)
            else:
                unambig_o.append(text)
            if (text_to_labels_combined[text] > 0).sum() > 1:
                ambig_c.append(text)
            else:
                unambig_c.append(text)
        logger.info(f"Processing {part_name} set...")
        logger.info(f"  Nb texts: {len(part_texts)}/{len(texts)}")
        logger.info(f"  Nb texts that are ambiguous based on ORIGINAL labels: {len(ambig_o)}/{len(part_texts)}")
        logger.info(f"  Nb texts that are unambiguous based on ORIGINAL labels: {len(unambig_o)}/{len(part_texts)}")
        logger.info(f"  Nb texts that are ambiguous based on COMBINED labels: {len(ambig_c)}/{len(part_texts)}")
        logger.info(f"  Nb texts that are unambiguous based on COMBINED labels: {len(unambig_c)}/{len(part_texts)}")

        # Write various datasets for this part
        for set_name, set_texts_o, set_texts_c in [("Ambig", ambig_o, ambig_c),
                                                   ("Unambig", unambig_o, unambig_c),
                                                   ("All", part_texts, part_texts)]:
            for mode in ["Single", "Single-no-in-class-dups", "Multi"]:
                subdir = os.path.join(args.dir_output, f"Original-labels/{set_name}/{mode}")
                write_data(set_texts_o, text_to_labels, classes, subdir, part_name, mode.lower())
                subdir = os.path.join(args.dir_output, f"Combined-labels/{set_name}/{mode}")
                write_data(set_texts_c, text_to_labels_combined, classes, subdir, part_name, mode.lower())
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_texts", help="Path of text file containing texts")
    p.add_argument("path_labels", help="Path of text file containing labels")
    p.add_argument("path_pickle", help="Path of pickle file containing similarity matrix and list of unique texts")
    p.add_argument("dir_output", help="Path of output directory")
    p.add_argument("--min_sim", "-m", type=float, default=0.9, help="minimum similarity for a text pair to be considered near duplicates")
    p.add_argument("--no_ne", "-e", action="store_true", help="Discard texts that contain mainly $NE$ tokens")
    p.add_argument("--nb_singles", "-b", type=int, help="Nb single-label examples to keep")
    p.add_argument("--train_prop", "-t", type=float, default=0.8, help="Proportion of unique texts to put in training set")
    p.add_argument("--dev_prop", "-d", type=float, default=0.1, help="Proportion of unique texts to put in dev set")
    p.add_argument("--seed", "-s", help="Seed for RNG")
    args = p.parse_args()
    main(args)
