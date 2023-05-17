import argparse
from sklearn.metrics import f1_score
from utils import load_lines, CLASS_NAMES

DOC="""
Evaluate predictions against gold labels.
"""

def main(args):
    # Load gold and predicted labels
    print("Loading data...")
    pred = [x.split(" ") if len(x) else [] for x in load_lines(args.path_pred)]
    gold = [x.split(" ") for x in load_lines(args.path_gold)]
    assert len(pred) == len(gold)
    for pl in pred:
        if not len(pl):
            assert args.problem_type == "multi", "Found an empty prediction, but this should not occur for single-label problem type."
            continue
        for p in pl:
            if p not in CLASS_NAMES:
                msg = f"Unrecognized class name '{p}' found in predicted label list '{pl}'"
                raise AssertionError(msg)
    for pl in gold:
        for p in pl:
            if p not in CLASS_NAMES:
                msg = f"Unrecognized class name '{p}' found in gold labels"
                raise AssertionError(msg)
    if any(len(x) > 1 for x in pred):
        assert args.problem_type == "multi", "problem_type must be 'multi' (some examples have more than one predicted label)"
    if any(len(x) > 1 for x in gold):
        assert args.problem_type == "multi", "problem_type must be 'multi' (some examples have more than one gold label)"

    # Compute evaluation metrics
    print("Computing evaluation metrics...")    
    if args.problem_type == "multi":
        gold_multi = [[1 if x in labels else 0 for x in CLASS_NAMES] for labels in gold]
        pred_multi = [[1 if x in labels else 0 for x in CLASS_NAMES] for labels in pred]        
        scores = f1_score(gold_multi, pred_multi, average=None, zero_division="warn")
        weighted = f1_score(gold_multi, pred_multi, average="weighted", zero_division="warn")
    else:
        label2id = {x:i for i,x in enumerate(CLASS_NAMES)}        
        gold_ids = [label2id[x[0]] for x in gold]
        pred_ids = [label2id[x[0]] for x in pred]        
        scores = f1_score(gold_ids, pred_ids, labels=range(len(label2id)), average=None, zero_division="warn")
        weighted = f1_score(gold_multi, pred_multi, average="weighted", zero_division="warn")

    # Print scores
    print()
    for i, score in enumerate(scores):
        print(f"F1 of class {CLASS_NAMES[i]}: {score:.5f}")
    mean_score = sum(scores) / len(scores)
    print(f"Macro-averaged F1: {mean_score:.5f}")
    print(f"Weighted average: {weighted:.5f}")
    print()
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_pred", help="Path of text file containing predicted labels")
    p.add_argument("path_gold", help="Path of text file containing gold labels")
    p.add_argument("problem_type", choices=["single", "multi"], help="Single label or multi label evaluation?")
    args = p.parse_args()
    main(args)
