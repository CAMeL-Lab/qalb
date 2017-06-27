# AI

Personal repository for managing [TensorFlow](https://www.tensorflow.org/) experiments.

## Directory structure

The repository's internal structure looks as follows:
	
	.
	├── ai/
	│   ├── datasets/
	│   │   └── data/ (ignored)
	│   ├── models/
	│   ├── tests/
	│   └── utils.py
	└── output/ (ignored)

The project is kept inside the `ai` folder to make it work as a module. This way, absolute imports can be done within any of the files of the project with ease, and the code can be organized efficiently.

For more information on each of the submodules, see the docstrings of their respective `__init__.py` files.

## Testing

For running created tests, run `python -m ai.tests.filename` in the top-level directory. See the individual test files for more information on the flags that can be passed to them directly via the terminal.

## Style guide

All python code should ahere to PEP-8 standards. Indentation is the only exception-- all the code in the project uses 2 spaces. All lines should not exceed 80 characters for readability.

As a convention for model classes, hyperparameters should always be set as keyword arguments for readability and to avoid issues with order. Since by nature there can be many hyperparameters for a single architecture, pylint is set to ignore when classes have too many attributes or methods have too many arguments.

The `lint.sh` file has a script to run pylint over all the project's files. To run it, use `bash lint.sh`. Alternatively, running `./lint.sh` is possible if the permissions to execute the file are set. This can be done by running the following command (and similarly for all the other executables):
	
	chmod u+rxw lint.sh

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
