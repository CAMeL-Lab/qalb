"""This module takes care of any preprocessing for every dataset. This way,
   the models can be kept as independent as possible from the input data. The
   `data` directory is ignored to not upload large files to the repository, and
   is expected to contain all the data files that the python files will read
   and preprocess.
   
   As a convention, all the different datasets are classes that can be initialized with different attributes, each living on its own file
   (including abstract classes) unless they are closely related. Thus, to make the imports less redundant, the classes are imported here directly, rather
   than adding the file names to the `__all__` variable as usual."""

from ai.datasets.base_dataset import BaseDataset
from ai.datasets.qalb import CharQALB, WordQALB
from ai.datasets.text_file import TextFile
from ai.datasets.trump_tweets import TrumpTweets
from ai.datasets.whatsapp_chats import WhatsAppChats
