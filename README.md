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

The `ai/datasets` directory contains python files that do any preprocessing to any dataset. This way, the models can be kept as independent as possible from the input data. The `ai/datasets/data` directory is ignored to not upload large files to the repository. It is expected to contain all the data files that the python files will read and preprocess.

As a convention, all the different datasets are classes that can be initialized with different attributes, each living on its own file (including abstract classes). Thus, to make the imports less redundant, the following line should be added to `ai/datasets/__init__.py`:
	
	from dataset_file import DatasetFile

This way, other modules can import with `from ai.datasets import DatasetFile`, rather than `from ai.datasets.dataset_file import DatasetFile`.

### Models

The `ai/models` directory contains all the python files that define different model computational graphs, as well as any of their abstractions. These should be independent to the data inputs. The same convention for module exports is used in `ai/models/__init__.py` as in `ai/datasets/__init__.py`.

### Tests

The `ai/tests` directory contains python scripts meant to execute experiments, putting together datasets and models. Besides abstractions or wrappers (which eventually will be made but don't yet exist), test files shouldn't be used by any other files, so nothing is done to `ai/tests/__init__.py`.

For running created tests, use `python -m ai.tests.filename`.
