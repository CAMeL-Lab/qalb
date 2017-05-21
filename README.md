# AI

Personal repository for managing [TensorFlow](https://www.tensorflow.org/) experiments.

## Directory structure

The repository's internal structure looks as follows:
	
	.
	├── ai
	│   ├── datasets
	│   │   └── data (ignored)
	│   ├── models
	│   └── tests
	└── output (ignored)

The project is kept inside the `ai` folder to make it work as a module. This way, absolute imports can be done within any of the files of the project with ease, and the code can be organized efficiently.

### Datasets

The `ai/datasets` module takes care of any preprocessing for every dataset. This way, the models can be kept as independent as possible from the input data. The `ai/datasets/data` directory is ignored to not upload large files to the repository. It is expected to contain all the data files that the python files will read and preprocess.

As a convention, all the different datasets are classes that can be initialized with different attributes, each living on its own file (including abstract classes) unless they are closely related. Thus, to make the imports less redundant, the following line should be added to `ai/datasets/__init__.py` for every class in each file:
	
	from ai.datasets.dataset_file import DatasetClass

This way, other modules can import with `from ai.datasets import DatasetFile`, rather than `from ai.datasets.dataset_file import DatasetFile`.

### Models

The `ai/models` module contains all the different model computational graphs, as well as any of their abstractions. These should be independent to the data inputs. The same conventions for module exports and class isolation to their own file are used in `ai/models/__init__.py` as in `ai/datasets/__init__.py`.

### Tests

The `ai/tests` module contains python scripts meant to execute experiments, putting together datasets and models. Besides abstractions or wrappers (which eventually will be made but don't yet exist), test files shouldn't be used by any other files, so nothing is done to `ai/tests/__init__.py`.

For running created tests, use `python -m ai.tests.filename`.

### Utils

The `ai/utils` lives in a single file (which will be modularized if it ever becomes too large), and is meant to provide handy methods that are not class dependent and overall useful.

## Style guide

All python code should ahere to PEP-8 standards. Indentation is the only exception-- all the code in the project uses 2 spaces. All lines should not exceed 80 characters for readability.

As a convention for model classes, hyperparameters should always be set as keyword arguments for readability. Since by nature there can be many hyperparameters for a single architecture, pylint is set to ignore when classes have too many attributes or methods have too many arguments.

The `lint.sh` file has a script to run pylint over all the project's files. To run it, use `bash lint.sh`. Alternatively, running `./lint.sh` is possible if the permissions to execute the file are set. This can be done by running the following command:
	
	chmod u+rxw lint.sh

