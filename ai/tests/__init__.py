"""This module contains python scripts meant to execute experiments, putting
   together datasets and models. Besides abstractions or wrappers (which
   eventually will be made but don't yet exist), test files shouldn't be used
   by any other files, so nothing should done to this file as of now.
   
   As a convention, all the tests should make use of the `tf.app.flags` methods
   provided by TensorFlow to allow changing hyperparameters and other config
   via the terminal when running the files."""
