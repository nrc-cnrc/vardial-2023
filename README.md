# Multi-label Dialect Identification

Experimental code on multi-label dialect identification, developed for the paper [Dialect and Variant Identification as a Multi-Label Classification Task: A Proposal Based on Near-Duplicate Analysis](https://aclanthology.org/2023.vardial-1.15/) (Bernier-Colborne, Goutte, and Léger; VarDial 2023).

We provide this code for the purpose of reproducing the experiments we conducted on the [FreCDo](https://github.com/MihaelaGaman/FreCDo) dataset. It is licensed under GPL 3.0, as it uses a library licensed under a previous version of GPL.

## Requirements

The scripts below require [Python](https://www.python.org/) (tested with version 3.9.12), and the following libraries (tested versions are in brackets):

- [NumPy](https://numpy.org/) (v 1.22.3)
- [SciPy](https://scipy.org/) (v 1.10.0)
- [scikit-learn](https://scikit-learn.org/stable/) (v 1.2.1)
- [PyTorch](https://pytorch.org/) (v 1.12.0)
- [Transformers](https://huggingface.co/docs/transformers/index) (v 4.20.1) 
- [Datasets](https://huggingface.co/docs/datasets/index) (v 2.3.2)
- [Levenshtein](https://github.com/maxbachmann/Levenshtein) (v 0.20.9)
- [tqdm](https://github.com/tqdm/tqdm) (v 4.64.0)


## Usage

The following commands assume that the text files containing the data are split into texts and labels, e.g.:

```bash
data/
	train.txt
	train.labels
	dev.txt
	dev.labels
	test.txt
	test.labels
```

This is the format produced by `make_dataset.py` (see below), but if you want to apply these commands to the original version of the FreCDo dataset, you will have to split the train and dev sets into separate files for texts and labels.

All the scripts mentioned below have their own internal documentation, so run `python <script-name> -h` for more details on usage.

To analyse exact duplicates in the data, use:

```bash
python count_dups.py data.txt data.labels
python show_dups.py data.txt data.labels
```

To analyse near-duplicates in the data using the Levenshtein edit ratio as similarity measure, with a cutoff at 0.8, use:

```bash
python make_sim_matrix.py data.txt sim.pkl -c 0.8 -b 1024 -p loky
python count_near_dups.py sim.pkl data.txt data.labels -m 0.8 -w log.txt -n token 
```

where `sim.pkl` will contain the result of the first command.

To make a random split from the original split of the FreCDo dataset, optionally combine labels of (near) duplicates, and produce various representations of the resulting data, use:

```bash
python make_dataset.py original-data.txt original-data.labels sim.pkl dir_modified_data -m 0.8 -t 0.85 -d 0.05
```

where `original-data.txt` and `original-data.labels` should contain the complete source data, and `dir_modified_data` will contain the result.

To finetune a [CamemBERT](https://huggingface.co/camembert-base) model and evaluate it, use one of the following (for single-label and multi-label classification respectively):

```bash
python finetune_single.py train.txt train.labels dev.txt dev.labels dir_checkpoint --freeze_embeddings --freeze_encoder_upto 10
python finetune_multi.py train.txt train.labels dev.txt dev.labels dir_checkpoint --freeze_embeddings --freeze_encoder_upto 10
```

where `dir_checkpoint` will contain the resulting model, the training logs, etc.

To evaluate classifiers, use:

```bash
python predict.py dir_checkpoint/checkpoint/best_model dir_checkpoint/checkpoint/tokenizer test.txt pred.labels
python evaluate.py pred.labels test.labels multi
```

where `pred.labels` will contain the predicted labels output by the first command.
