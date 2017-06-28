"""This module contains all the different model computational graphs, as well
   as any of their abstractions. These should be independent to the data
   inputs.

   As a convention, all the different datasets are classes that can be
   initialized with different attributes, each living on its own file
   (including abstract classes) unless they are closely related. Thus, to make
   the imports less redundant, the classes are imported here directly, rather
   than adding the file names to the `__all__` variable as usual."""

from ai.models.base_model import BaseModel
#from ai.models.dnc import DNC
from ai.models.next_gram import NextGram
from ai.models.seq2seq import Seq2Seq
from ai.models.softmax_embeddings import SoftmaxEmbeddings
