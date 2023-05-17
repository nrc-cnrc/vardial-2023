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

