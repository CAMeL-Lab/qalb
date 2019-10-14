# QALB

Code for [Utilizing Character and Word Embeddings for Text Normalization with Sequence-to-Sequence Models](https://www.aclweb.org/anthology/papers/D/D18/D18-1097/).

## Setup

Python2, Python3, and TensorFlow 1.4 are required to run this project.

## Training & testing

`python -m ai.tests.qalb` will run a generic character-level model. `python -m ai.tests.char_qalb` will run the hybrid character-level + fastText model. To distinguish between training and inference, `--decode=path/to/file.txt` indicates to run the model on inference mode on that text file.

See the individual test scripts for more information on the flags that can be passed to them directly via the terminal.

## Evaluations

To compute the F1 score, use `python2 ai/tests/m2scripts/m2scorer.py --beta 1 -v $1 $2` where `$1` is the system output file and `$2` is the `.m2` gold file.

To compute the Levenshtein score, use `python levenshtein.py $1 $2` where `$1` is the output file and `$2` is the `.gold` file.

## Analysis

`python analysis.py` can be used to break down any `.m2` file into a more human-readable format.

## Useful UNIX commands

To remove the document id's from the `*.sent*` files, simply use

	cut -d' ' -f2- ai/datasets/data/qalb/FILENAME

This can be piped to give a word count:

	# *.sent* file
	cut -d' ' -f2- ai/datasets/data/qalb/QALB.train.sent.sbw | awk '{print NF}'
	# *.gold* file
	cat ai/datasets/data/qalb/QALB.train.gold.sbw | awk '{print NF}'

Or a character count:

	cat ai/datasets/data/qalb/QALB.train.gold.sbw | awk '{ print length($0); }'

The total number of usable characters doesn't include newlines, so as an example, the number of characters in a `*.sent*` file can be obtained automatically with

	cut -d' ' -f2- ai/datasets/data/qalb/QALB.train.sent.sbw | awk '{ print length($0); }' | awk '{s+=$1} END {print s}'

Which coincides with doing `cut -d' ' -f2- ai/datasets/data/qalb/QALB.train.sent.sbw | wc` and subtracting the number of lines to the number of characters.

To obtain a histogram of the character counts, simply pipe `sort` and `uniq`. For instance,
	# Characters
	cat ai/datasets/data/qalb/QALB.train.gold.sbw | awk '{ print length($0); }' | sort -n | uniq -c
	# Words
	cat ai/datasets/data/qalb/QALB.train.gold.sbw | awk '{print NF}' | sort -n | uniq -c
