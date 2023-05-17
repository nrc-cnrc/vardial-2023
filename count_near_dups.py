import os, argparse, pickle, logging, random
from difflib import ndiff
from tqdm import tqdm
import numpy as np
from scipy.sparse import tril, find
from utils import load_lines

DOC="""
Count near-duplicates in dataset.
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Loading data from {args.path_pickle}...")
    with open(args.path_pickle, 'rb') as f:
        data = pickle.load(f)
        mat = data["matrix"]
        mat_labels = data["labels"]
    mat = mat.tocoo()
    logger.info(f"Type of matrix: {type(mat)}")
    logger.info(f"Shape of matrix: {mat.shape}")    
    logger.info(f"Nb nnz: {mat.nnz}")
    logger.info(f"Loading texts from {args.path_texts} and labels from {args.path_labels}...")
    texts = load_lines(args.path_texts)
    labels = load_lines(args.path_labels)
    assert len(texts) == len(labels)

    # Map texts to labels
    text_to_labels = {}
    for x,y in zip(texts, labels):
        if x not in text_to_labels:
            text_to_labels[x] = []
        text_to_labels[x].append(y)

    # Count near duplicates
    logger.info(f"Identifying near duplicates with sim>={args.min_sim}...")
    mat = tril(mat, k=1)
    rows, cols, vals = find(mat)
    nd = [(rows[i],cols[i],vals[i]) for i in np.where(vals>=args.min_sim)[0]]    
    logger.info("Counting near duplicates that have different sets of unique labels...")        
    ambig = set()
    pbar = tqdm(total=len(nd))
    for (i,j,s) in nd:
        li = set(text_to_labels[mat_labels[i]])
        lj = set(text_to_labels[mat_labels[j]])
        if len(li.symmetric_difference(lj)):
            ambig.add((i,j,s))
        pbar.update(1)
    pbar.close()
    logger.info(f"Nb near duplicate pairs: {len(nd)}")
    logger.info(f"Nb ambiguous duplicate pairs: {len(ambig)}/{len(nd)}")

    # Count frequency of edits, and format them for printing
    logger.info("Counting frequency of edits...")
    ambig = sorted(ambig, key=lambda x:x[2], reverse=True)
    edit2freq = {}
    pretty_ambig = []    
    pbar = tqdm(total=len(ambig))
    delim = " " if args.ndiff_type == "token" else ""
    for ix,(i,j,sim) in enumerate(ambig):
        # Do diff and format edit blocks
        ops = []        
        prev_op = None
        block = []
        if args.ndiff_type == "char":
            diff = ndiff(mat_labels[i],mat_labels[j])
        elif args.ndiff_type == "token":
            tokens_i = [x+" " for x in mat_labels[i].split(" ")]
            tokens_j = [x+" " for x in mat_labels[j].split(" ")]
            diff = ndiff(tokens_i, tokens_j)
        for x in diff:
            op = x[0]
            if op != prev_op:
                if len(block):
                    pretty_op = "=" if prev_op == " " else prev_op
                    pretty_block = delim.join(block)
                    ops.append((pretty_op, pretty_block))                    
                block = []
                prev_op = op
            block.append(x[2:])

        if len(block):
            pretty_op = "=" if prev_op == " " else prev_op
            pretty_block = delim.join(block)
            ops.append((pretty_op, pretty_block))
            
        # Store message for printing
        msg = f"************* Example {ix+1} ****************\n"
        msg += f"- Sim={sim:.5f}\n"
        msg += f"- Diff:\n"
        for (symbol, string) in ops:
            msg += f"  {symbol} [{string}]\n"
        msg += f"- Text {i}: {mat_labels[i]}\n"
        msg += f"- Labels of text {i}: {text_to_labels[mat_labels[i]]}\n"        
        msg += f"- Text {j}: {mat_labels[j]}\n"            
        msg += f"- Labels of text {j}: {text_to_labels[mat_labels[j]]}\n"            
        pretty_ambig.append(msg)

        # Concatenate edit ops, and count them
        block = []
        edits = []
        for (symbol, string) in ops:
            if symbol == "=":
                if len(block):
                    edits.append(tuple(block))
                    block = []
            else:
                block.append(symbol)
                block.append(string)
        if len(block): 
            edits.append(tuple(block))
        for edit in edits:
            if edit not in edit2freq:
                edit2freq[edit] = 0
            edit2freq[edit] += 1
        pbar.update(1)
    pbar.close()

    # Print most frequent edits
    logger.info("Most frequent edits:")
    k = 10
    topk = sorted(edit2freq.keys(), key=edit2freq.get, reverse=True)[:k]
    for i,edit in enumerate(topk):
        freq = edit2freq[edit]
        logger.info(f"  {i+1}. Freq={freq}")
        for i in range(0, len(edit)//2, 2):
            symbol = edit[i]
            string = edit[i+1]
            logger.info(f"    {symbol} [{string}]")
    
    # Write ambiguous near duplicates
    if args.write_to:
        if args.seed:
            random.seed(args.seed)
        if args.ndiff_sample_size:
            logger.info(f"Sampling {args.ndiff_sample_size} ndiff examples at random...")
            assert args.ndiff_sample_size < len(pretty_ambig)
            sample = np.random.choice(len(pretty_ambig), args.ndiff_sample_size, False)
            sample.sort()
            pretty_ambig = [pretty_ambig[i] for i in sample]
        logger.info(f"Writing ambiguous near duplicates to {args.write_to}...")
        with open(args.write_to, 'w') as f:
            for msg in pretty_ambig:
                f.write(msg + "\n")
    logger.info("Done.")
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_pickle", help="Path of pickle file containing similarity matrix and list of unique texts")
    p.add_argument("path_texts", help="Path of text file containing texts")
    p.add_argument("path_labels", help="Path of text file containing labels")
    p.add_argument("--min_sim", "-m", type=float, default=0, help="minimum similarity for a text pair to be considered near duplicates")
    p.add_argument("--write_to", "-w", type=str, help="Optional path of file to write ambiguous near duplicates to.")
    p.add_argument("--ndiff_type", "-n", choices=["char", "token"], default="char", help="Type of ndiff used to highligh differences (if --write_to is specified)")
    p.add_argument("--ndiff_sample_size", "-i", type=int, help="Number of ndiffs to sample (if --write_to is specified)")
    p.add_argument("--seed", "-s", help="Seed for RNG  (used for sampling ndiff outputs")    
    args = p.parse_args()
    assert args.min_sim >= 0
    assert args.min_sim < 1
    if args.write_to:
        assert not os.path.exists(args.write_to)
    if args.ndiff_sample_size:
        assert args.write_to
    if args.seed:
        assert args.write_to
    main(args)
