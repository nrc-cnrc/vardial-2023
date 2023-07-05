
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
from utils import load_lines

DOC="""
Show examples of texts that have duplicates in muliple classes.
"""

MIN_UNIQ_CLASSES = 2

def show_dups(texts, labels):
    text_to_labels = {}
    for text, label in zip(texts, labels):
        if text not in text_to_labels:
            text_to_labels[text] = []
        text_to_labels[text].append(label)
    dups_multi = [text for text, labels in text_to_labels.items() if len(set(labels)) >= MIN_UNIQ_CLASSES]
    print("\nTexts that have duplicates in multiple classes:")
    for i, text in enumerate(dups_multi):
        print(f"{i+1}. {text} [{', '.join(sorted(text_to_labels[text]))}]")
    print()
    return

def main(args):
    texts = load_lines(args.path_texts)
    labels = load_lines(args.path_labels)
    assert len(texts) == len(labels)
    show_dups(texts, labels)
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_texts", help="Path of text file containing texts")
    p.add_argument("path_labels", help="Path of text file containing labels")
    args = p.parse_args()
    main(args)
