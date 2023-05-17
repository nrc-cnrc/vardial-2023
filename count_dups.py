import os, argparse
from collections import Counter
from itertools import combinations
from utils import load_lines

DOC="""
Count exact duplicates in dataset. 
"""

def count_dups(texts, labels):
    data = list(zip(texts, labels))

    # Show label frequency distribution
    text_to_labels = {}
    label_fd = {}
    for (text, label) in data:
        if text not in text_to_labels:
            text_to_labels[text] = []
        text_to_labels[text].append(label)
        if label not in label_fd:
            label_fd[label] = 0
        label_fd[label] += 1
    label_set = sorted(label_fd.keys())
    print("\n------------------------")
    print(f"\nNb classes: {len(label_set)}")
    for label in label_set:
        print(f"  - {label} (n={label_fd[label]})") 
    print("\n------------------------")

    # Show distribution of # labels/text
    fd_labels_per_text = {}
    fd_uniq_labels_per_text = {}
    for (text, labels) in text_to_labels.items():
        if len(labels) not in fd_labels_per_text:
            fd_labels_per_text[len(labels)] = 0
        fd_labels_per_text[len(labels)] += 1
        if len(set(labels)) not in fd_uniq_labels_per_text:
            fd_uniq_labels_per_text[len(set(labels))] = 0
        fd_uniq_labels_per_text[len(set(labels))] += 1                
    print(f"\nDistribution of # labels per unique text:")
    for count in sorted(fd_labels_per_text.keys()):
        print(f"{count}: {fd_labels_per_text[count]}")
    print(f"\nDistribution of # unique labels per unique text:")
    for count in sorted(fd_uniq_labels_per_text.keys()):
        print(f"{count}: {fd_uniq_labels_per_text[count]}")
    print("\n------------------------")    
    
    # Show class-wise stats
    class2texts = {}
    class2texts['ALL'] = texts
    for (text, label) in data:
        if label not in class2texts:
            class2texts[label] = []
        class2texts[label].append(text)    
    for label in (label_set + ["ALL"]):
        texts = class2texts[label]
        print(f"\nStats for class '{label}':")
        print(f"- Nb texts: {len(texts)}")
        print(f"- Nb unique texts: {len(set(texts))}")
        ratio = len(set(texts))/len(texts)
        print(f"- Ratio: {ratio:.5f}")
        if label == 'ALL':
            classcount_fd = {}
            uclasscount_fd = {}
            nb_dups = 0
            unb_dups = 0
            for (text, labels) in text_to_labels.items():
                if len(labels) > 1:
                    nb_dups += len(labels)
                    unb_dups += 1
                    classcount = len(set(labels))
                    if classcount not in classcount_fd:
                        classcount_fd[classcount] = 0
                    classcount_fd[classcount] += len(labels)
                    if classcount not in uclasscount_fd:
                        uclasscount_fd[classcount] = 0
                    uclasscount_fd[classcount] += 1
            print(f"- Nb texts that have a duplicate in any class: {nb_dups}/{len(data)}")
            print(f"- Nb unique texts that have a duplicate in any class: {unb_dups}/{len(text_to_labels)}")            
            for classcount in sorted(classcount_fd.keys()):
                freq = classcount_fd[classcount]
                ufreq = uclasscount_fd[classcount]                
                print(f"- Nb texts that have duplicates within {classcount} class(es): {freq}/{len(data)}")
                print(f"- Nb unique texts that have duplicates within {classcount} class(es): {ufreq}/{len(text_to_labels)}")
        else:
            dupcount_inclass = 0
            dupcount_outclass = 0
            dupcount_anyclass = 0
            udupcount_inclass = 0
            udupcount_outclass = 0
            udupcount_anyclass = 0
            for text in set(texts):
                labels = text_to_labels[text]
                assert label in labels
                label_fd = Counter(labels)
                if label_fd[label] > 1:
                    dupcount_inclass += label_fd[label]
                    udupcount_inclass += 1
                if len(label_fd) > 1:
                    dupcount_outclass += label_fd[label]
                    udupcount_outclass += 1
                if len(labels) > 1:
                    dupcount_anyclass += label_fd[label]
                    udupcount_anyclass += 1
            print(f"- Nb texts that have a duplicate within this class: {dupcount_inclass}/{len(texts)}")
            print(f"- Nb texts that have a duplicate in another class: {dupcount_outclass}/{len(texts)}")
            print(f"- Nb texts that have a duplicate in any class: {dupcount_anyclass}/{len(texts)}")        
            print(f"- Nb unique texts that have a duplicate within this class: {udupcount_inclass}/{len(set(texts))}")
            print(f"- Nb unique texts that have a duplicate in another class: {udupcount_outclass}/{len(set(texts))}")
            print(f"- Nb unique texts that have a duplicate in any class: {udupcount_anyclass}/{len(set(texts))}")        
    print("\n------------------------\n")

    # Show most frequently confused label pairs
    pair_to_count = {}
    for text, labels in text_to_labels.items():
        if len(set(labels)) > 1:
            for pair in combinations(set(labels), 2):
                pair = tuple(sorted(pair))
                if pair not in pair_to_count:
                    pair_to_count[pair] = 0
                pair_to_count[pair] += 1
    print("\nMost frequently confused pairs of labels:")
    for (pair, count) in sorted(pair_to_count.items(), key=lambda x:x[1], reverse=True):
        x, y = pair
        print(f"{x} {y}: {pair_to_count[pair]}")

    print("\n------------------------\n")    
    return

def main(args):
    texts = load_lines(args.path_texts)
    labels = load_lines(args.path_labels)
    assert len(texts) == len(labels)
    count_dups(texts, labels)
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_texts", help="Path of text file containing texts")
    p.add_argument("path_labels", help="Path of text file containing labels")
    args = p.parse_args()
    main(args)
