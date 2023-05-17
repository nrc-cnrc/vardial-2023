import os, argparse, pickle, logging, multiprocessing, random
from copy import deepcopy
from math import ceil
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.sparse import lil_matrix, csr_matrix
from Levenshtein import ratio
from utils import load_lines

DOC="""
Compute pairwise similarity matrix for a set of texts. 
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vectors(path, order=None, skip_head=True):
    """Load feature vectors. Expects each line to contain the class ID,
    followed by (feature ID, count) pairs, followed by a single-word
    comment, all space-separated.

    Args:
    - path
    - order: list of text IDs to read and add to matrix
    - skip_head: skip header?
    """
    
    # Get number of texts and check order
    if order is None:
        nb_texts = sum(1 for line in open(path))
        order = range(nb_texts)
    text_id_to_row_id = {}
    for i, text_id in enumerate(order):
        text_id_to_row_id[text_id] = i

    # Parse feature vectors
    dim = 0
    vecs = [None] * len(text_id_to_row_id)    
    with open(path) as f:
        # Skip header
        if skip_head:
            line = f.readline()

        # Parse rest of file
        for text_id, line in enumerate(f):
            if text_id not in text_id_to_row_id:
                continue
            elems = line.rstrip().split(" ")
            assert elems[-2] == "#"  # Last 2 elements are a comment
            feats = elems[1:-2] # Remove class and comment
            vec = []
            for f in feats:
                e = f.split(":")
                assert len(e) == 2
                feat = int(e[0]) 
                val = int(e[1])
                if feat > dim:
                    dim = feat
                vec.append((feat, val))
            vecs[text_id_to_row_id[text_id]] = vec
    matrix = lil_matrix((len(text_id_to_row_id), dim), dtype='float')
    for i,vec in enumerate(vecs):
        for (feat, val) in vec:
            feat = feat - 1 # Subtract 1 because feature IDs are 1-indexed
            matrix[i,feat] = val
    return matrix

def compute_vector_sim(args):
    # Get texts
    texts = load_lines(args.path_texts)
    utext_to_ids = {}
    for i,text in enumerate(texts):
        if text not in utext_to_ids:
            utext_to_ids[text] = []
        utext_to_ids[text].append(i)

    # Sort by length
    uniq_texts = sorted(utext_to_ids.keys(), key=lambda x:len(x))
    logger.info(f"Nb texts: {len(texts)}")
    logger.info(f"Nb unique texts: {len(uniq_texts)}")
    dim = len(uniq_texts)    
    if args.max_k_per_row:
        assert args.max_k_per_row < dim
    nb_comp = (dim ** 2 - dim) // 2
    logger.info(f"Nb pairs of texts to compair: {nb_comp}")
    
    # Get feature vectors
    logger.info(f"Loading feature vectors from {args.vecs}")
    order = [utext_to_ids[k][0] for k in uniq_texts]
    vecs = load_vectors(args.vecs, order)
    logger.info("Converting matrix to CSR for more efficient operations...")
    vecs = vecs.tocsr()
    logger.info(f"Shape of feature vectors: {vecs.shape}")    
    avg_feats = vecs.nnz / len(texts)
    nb_cells = vecs.shape[0] * vecs.shape[1]
    sparsity = 100 * ((nb_cells-vecs.nnz)/nb_cells)
    nb_zero_vecs = 0
    for i in range(dim):
        if vecs[i].sum() == 0:
            nb_zero_vecs += 1
        else:
            break
    logger.info(f"Nb non-zero elements: {vecs.nnz} (average={avg_feats:.1f}/text, sparsity={sparsity:.5f}%)")
    logger.info(f"Nb zero vectors: {nb_zero_vecs}")
    
    # Compute similarities
    logger.info(f"Initializing sparse sim matrix...")    
    matrix = lil_matrix((dim,dim), dtype='float')
    row_order = list(range(dim))
    random.shuffle(row_order)
    nb_batches = ceil(dim / args.batch_size)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Nb batches: {nb_batches}")
    logger.info(f"Computing pairwise similarities...")
    pbar = tqdm(total=nb_batches)
    for i in range(nb_batches):
        batch_ixs = row_order[i*args.batch_size:(i+1)*args.batch_size]
        batch = vecs[batch_ixs,:]
        if args.sim_measure == "cosine":
            sim_batch = cosine_similarity(batch, vecs, dense_output=True)
        elif args.sim_measure == "manhattan":
            sim_batch = 1 - (manhattan_distances(batch, vecs) / (batch.sum(1) + vecs.sum(1).reshape(1,-1)))
        if args.max_k_per_row:
            ctz = np.argsort(sim_batch)[:,:-args.max_k_per_row]
            rtz = np.arange(sim_batch.shape[0], dtype=int)[:,np.newaxis] * np.ones(ctz.shape, dtype=int)
            sim_batch[rtz,ctz] = 0
        sim_batch = lil_matrix(sim_batch)
        matrix[batch_ixs,:] = sim_batch
        pbar.update(1)
    pbar.close()
    return matrix, uniq_texts

def batched_sim(text_pairs, cutoff):
    results = []
    for (t1,t2) in text_pairs:
        sim = ratio(t1, t2, score_cutoff=cutoff)
        results.append(sim)
    return results

def compute_edit_ratio(args):
    texts = load_lines(args.path_texts)
    uniq_texts = list(set(texts))
    logger.info(f"Nb texts: {len(texts)}")
    logger.info(f"Nb unique texts: {len(uniq_texts)}")
    dim = len(uniq_texts)    
    nb_comp = (dim ** 2 - dim) // 2
    logger.info(f"Nb pairs of texts to compair (w/o pre-filtering by length diff): {nb_comp}")

    # Init matrix and set diag to 1
    logger.info(f"Initializing sparse sim matrix and setting diagonal to 1...")    
    matrix = lil_matrix((dim,dim), dtype='float')
    for i in range(dim):
        matrix[i,i] = 1.0

    # Sort texts by length
    uniq_texts = sorted(uniq_texts, key=lambda x:len(x))    
    lens = [len(x) for x in uniq_texts]
    row_order = list(range(len(uniq_texts)))
    random.shuffle(row_order)
    
    # Compute Levenshtein similarities
    nb_cores = multiprocessing.cpu_count()
    nb_batches = int(ceil(len(uniq_texts) / args.batch_size))
    logger.info(f"Parallel backend: {args.parallel_backend}")
    logger.info(f"Nb CPU cores available: {nb_cores}")
    logger.info(f"Batch size: {args.batch_size} rows")
    logger.info(f"Nb batches: {nb_batches}")
    logger.info(f"Computing pairwise similarity scores...")
    pbar = tqdm(total=nb_batches)    
    for batch_id in range(nb_batches):
        # Get batch of pairs of texts
        batch_ixs = []        
        rows = row_order[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
        for row in rows:
            limit = ((2-args.cutoff) * lens[row])/args.cutoff
            for col in range(row+1):
                if lens[col] > limit:
                    break
                else:
                    batch_ixs.append((row,col))
        random.shuffle(batch_ixs)
        batch = [(uniq_texts[i], uniq_texts[j]) for (i,j) in batch_ixs]

        # Check if we need to do embarassingly parallel processing
        if args.parallel_backend == 'none':
            sims = [ratio(t1, t2, score_cutoff=args.cutoff) for (t1,t2) in batch]
        else:
            step = len(batch) / nb_cores
            bin_edges = [(round(step*i),round(step*(i+1))) for i in range(nb_cores)]
            inputs = [(batch[start:stop], args.cutoff) for (start,stop) in bin_edges]

            # Compute similarities
            with parallel_backend(backend=args.parallel_backend, n_jobs=nb_cores):
                with Parallel() as parallel:
                    results = parallel(delayed(batched_sim)(*x) for x in inputs)
                
            # Flatten results
            sims = [inner for outer in results for inner in outer]
        
        # Put results in matrix
        for ((row,col), sim) in zip(batch_ixs, sims):
            if sim > 0:
                matrix[row,col] = sim
        pbar.update(1)
    pbar.close()
    return matrix, uniq_texts

def main(args):
    if args.vecs:
        matrix, uniq_texts = compute_vector_sim(args)
    else:
        matrix, uniq_texts = compute_edit_ratio(args)

    # Finish up
    logger.info("Converting matrix to CSR for more efficient operations...")
    matrix = matrix.tocsr()
    logger.info(f"Pickling -> {args.path_output}")
    data = {}
    data["matrix"] = matrix
    data["labels"] = uniq_texts
    with open(args.path_output, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Done.\n")
    return

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=DOC)
    p.add_argument("path_texts", help="Path of text file containing texts")
    p.add_argument("path_output", help="Path of output file a pickle file containing a sparse sim matrix and the list of unique texts which correspond to the rowa and columns")
    p.add_argument("--vecs", "-v", help="Path of feature vectors, one per input text. If not provided, we use Levenshtein ratios instead of vector similarity")
    p.add_argument("--cutoff", "-c", type=float, default=0.9, help="Min threshold on the Levenshtein ratio, under which it will be set to 0")
    p.add_argument("--max_k_per_row", "-m", type=int, help="Maximum nb of top similarities to keep per row (when using vector-based similarity)")
    p.add_argument("--sim_measure", "-s", choices=["manhattan", "cosine"], default="manhattan", help="Similarity measure (--vecs must be provided)")
    p.add_argument("--batch_size", "-b", type=int, default=1000, help="Batch size (nb rows) for computing pairwise similarities")
    p.add_argument("--parallel_backend", "-p", choices=['none', 'loky', 'multiprocessing', 'threading'], default='none', help="Backend for parallel processing (if using Levenshtein similarity)")
    args = p.parse_args()
    assert not os.path.exists(args.path_output)
    assert args.cutoff > 0
    assert args.cutoff <= 1.0
    assert args.batch_size > 0
    if args.max_k_per_row:
        assert args.max_k_per_row > 0
    main(args)
