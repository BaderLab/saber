"""Contains the Saber class, which exposes most of Sabers functionality.
"""
import logging
import os
import pickle
import time
from itertools import chain
from pprint import pprint

from spacy import displacy

from . import constants
from .config import Config
from .dataset import Dataset
from .embeddings import Embeddings
from .preprocessor import Preprocessor
from .trainer import Trainer
from .utils import data_utils, generic_utils, grounding_utils, model_utils

print('Saber version: {0}'.format(constants.__version__))
LOGGER = logging.getLogger(__name__)


class Saber(object):
    """The interface for Saber.

    As the interface to Saber, this class exposes all of Sabers functionality, including the
    training, saving, and loading of sequence labelling models.

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line. If not provided, a new instance of
            Config is used.
    """
    def __init__(self, config=None, **kwargs):
        self.config = Config() if config is None else config

        self.preprocessor = None # object for text processing
        self.datasets = None # dataset(s) tied to this instance
        self.embeddings = None # pre-trained token embeddings tied to this instance
        self.model = None # model object tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.config.verbose:
            print('Hyperparameters and model details:')
            pprint({arg: getattr(self.config, arg) for arg in constants.CONFIG_ARGS})
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    def annotate(self, text, model_idx=0, jupyter=False, coref=False, ground=False):
        """Uses a trained model (`self.model`) to annotate `text`, returns a dictionary.

        For the model at `self.model.models[model_idx]`, coordinates a prediction step on `text`.
        Returns a dictionary containing the cleaned text (`text`), and any annotations made by the
        model (`ents`). If `jupyter` is True, renders a HTMl visualization of the annotations made
        by the model, for use in a jupyter notebook.

        text (str): Raw text to annotate.
        model_idx (int): Index of model to use for prediction, defaults to 0.
        title (str): Title of the document, defaults to None.
        coref (book): True if coreference resolution should be performed before annotation, defaults
            to False.
        jupyter (bool): True if annotations made by the model should be rendered in HTML, which can
            be visualized in a jupter notebook, defaults to false.

        Returns:
            Dictionary containing the processed input text (`text`) and any annotations made by the
            model (`ents`).

        Raises:
            ValueError: If `text` is invalid (not a string, or empty/falsey).
        """
        # this takes about a minute to load, so only do it once!
        if self.preprocessor is None:
            self.preprocessor = Preprocessor()

        if not isinstance(text, str) or not text:
            err_msg = "Argument `text` must be a valid, non-empty string."
            LOGGER.error("ValueError: %s", err_msg)
            raise ValueError(err_msg)

        # model and its corresponding dataset
        ds = self.datasets[model_idx]
        model = self.model.models[model_idx]

        # process raw input text, collect input to ML model
        transformed_text = self.preprocessor.transform(text=text,
                                                       word2idx=ds.type_to_idx['word'],
                                                       char2idx=ds.type_to_idx['char'],
                                                       coref=coref)
        model_input = [transformed_text['word_idx_seq'], transformed_text['char_idx_seq']]
        # perform prediction, convert from one-hot to predicted indices, flatten results
        y_pred = model.predict(model_input, constants.PRED_BATCH_SIZE).argmax(-1).ravel()
        # convert predictions to tags (removing pads) and then chunk
        pred_tag_seq = [ds.idx_to_tag[idx] for idx in y_pred if ds.idx_to_tag[idx] != constants.PAD]
        pred_chunk_seq = self.preprocessor.chunk_entities(pred_tag_seq)
        # flatten the token offsets
        offsets = list(chain.from_iterable(transformed_text['offsets']))

        # accumulate predicted entities
        ents = []
        for chunk in pred_chunk_seq:
            # chunks are tuples (label, start, end), offsets is a lists of lists of tuples
            label, start, end = chunk[0], offsets[chunk[1]][0], offsets[chunk[-1] - 1][-1]
            text = transformed_text['text'][start:end]
            ents.append({'start': start, 'end': end, 'text': text, 'label': label})

        # create the final annotation and ground it
        annotation = {'text': transformed_text['text'], 'ents': ents}
        if ground:
            annotation = grounding_utils.ground(annotation)

        if jupyter:
            displacy.render(annotation, jupyter=True, style='ent', manual=True,
                            options=constants.OPTIONS)

        return annotation

    def save(self, directory=None, compress=True, model_idx=0):
        """Coordinates the saving of Saber models.

        Saves the necessary files for model persistence to `directory`. If not provided, `directory`
        defaults to '<self.config.output_folder>/<constants.PRETRAINED_MODEL_DIR>/<dataset_names>'

        Args:
            directory (str): Directory path to save model to. If None, a default directory under
                `self.config.output_folder` is created.
            compress (bool): True if model should be saved as tarball. Defaults to True.
            model_idx (int): Which model in `self.model.models` to save, defaults to 0.
        """
        start = time.time()
        print('Saving model...', end=' ', flush=True)

        # allows user to call save without providing a directory, default one is chosen
        if directory is None:
            directory = model_utils.prepare_pretrained_model_dir(self.config)

        directory = generic_utils.clean_path(directory)
        generic_utils.make_dir(directory)

        weights_filepath = os.path.join(directory, constants.WEIGHTS_FILENAME)
        model_filepath = os.path.join(directory, constants.MODEL_FILENAME)
        self.model.save(weights_filepath, model_filepath, model_idx)
        # beside the architecture and weights, save only the objects we need for model prediction
        attributes_filepath = os.path.join(directory, constants.ATTRIBUTES_FILENAME)
        model_attributes = {'model_name': self.config.model_name,
                            'type_to_idx': self.datasets[model_idx].type_to_idx,
                            'idx_to_tag': self.datasets[model_idx].idx_to_tag}
        pickle.dump(model_attributes, open(attributes_filepath, 'wb'))
        # save config for reproducibility
        self.config.save(directory)

        if compress:
            generic_utils.compress_directory(directory)

        end = time.time() - start
        print('Done ({0:.2f} seconds).'.format(end))
        print('Model was saved to {}'.format(directory))
        LOGGER.info('Model was saved to %s', directory)

    def load(self, directory):
        """Coordinates the loading of Saber models.

        Loads a Saber model saved at `directory`. Creates and compiles a new model with identical
        architecture and weights.
        """
        start = time.time()
        print('Loading model...', end=' ', flush=True)

        # Allows user to provide names of pre-trained models (e.g. 'PRGE') rather than filepaths
        if directory.upper() in constants.PRETRAINED_MODELS:
            directory = os.path.join(constants.PRETRAINED_MODEL_DIR, directory.upper())

        directory = generic_utils.clean_path(directory)
        generic_utils.extract_directory(directory)

        # load only the objects we need for model prediction
        attributes_filepath = os.path.join(directory, constants.ATTRIBUTES_FILENAME)
        model_attributes = pickle.load(open(attributes_filepath, "rb"))
        # its easiest to use a Dataset object to hold these data objects
        self.datasets = [Dataset()]
        self.datasets[0].type_to_idx = model_attributes['type_to_idx']
        self.datasets[0].idx_to_tag = model_attributes['idx_to_tag']
        # prevents user from having to specify pre-trained models name
        # TEMP: the get statement is for older models that didn't save this attribute, will
        # remove when those models are gone
        self.config.model_name = model_attributes.get('model_name', self.config.model_name)

        # create new instance of model, load pre-trained weights
        weights_filepath = os.path.join(directory, constants.WEIGHTS_FILENAME)
        model_filepath = os.path.join(directory, constants.MODEL_FILENAME)
        self.model = model_utils.load_pretrained_model(config=self.config,
                                                       datasets=self.datasets,
                                                       weights_filepath=weights_filepath,
                                                       model_filepath=model_filepath)

        end = time.time() - start
        print('Done ({0:.2f} seconds).'.format(end))

    def load_dataset(self, directory=None):
        """Coordinates the loading of a dataset.

        Args:
            directory (str): Path to a dataset folder. If not None, overwrites
                `self.config.dataset_folder`

        Raises:
            ValueError: If `self.config.dataset_folder` is None and `directory` is None.
        """
        start = time.time()
        # allows a user to provide the dataset directory when function is called
        if directory is not None:
            directory = directory if isinstance(directory, list) else [directory]
            directory = [generic_utils.clean_path(dir_) for dir_ in directory]
            self.config.dataset_folder = directory

        if not self.config.dataset_folder:
            err_msg = "Must provide at least one dataset via the `dataset_folder` parameter"
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        # need to save the type to index mapping here for setting up transfer learning
        type_to_idx = None if self.datasets is None else self.datasets[0].type_to_idx

        # datasets may be 'single' or 'compound' (more than one)
        if len(self.config.dataset_folder) == 1:
            print('Loading (single) dataset...', end=' ', flush=True)
            self.datasets = data_utils.load_single_dataset(self.config)
            LOGGER.info('Loaded single dataset at: %s', self.config.dataset_folder)
        else:
            print('Loading (compound) dataset...', end=' ', flush=True)
            self.datasets = data_utils.load_compound_dataset(self.config)
            LOGGER.info('Loaded multiple datasets at: %s', self.config.dataset_folder)

        # if a model has already been loaded, assume we are transfer learning
        if self.model is not None:
            print('\nTransferring from pre-trained model...', end=' ', flush=True)
            LOGGER.info('Transferred from a pre-trained model')
            # make required changes to target datasets
            for dataset in self.datasets:
                data_utils.setup_dataset_for_transfer(dataset, type_to_idx)
            # prepare model for transfer learning
            self.model.prepare_for_transfer(self.datasets)

        end = time.time() - start
        print('Done ({0:.2f} seconds).'.format(end))

    def load_embeddings(self, filepath=None, binary=True, load_all=None):
        """Coordinates the loading of pre-trained token embeddings.

        Args:
            filepath (str): Path to pre-trained embeddings file. If not None, overwrites
                'self.config.pretrained_embeddings'.
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format.
            load_all (bool): True if all pre-trained embeddings should be loaded. Otherwise only
                embeddings for tokens found in the training set are loaded. Defaults to None.

        Raises:
            MissingStepException: If no dataset has been loaded.
            ValueError: If 'self.config.pretrained_embeddings' is None and `filepath` is None.
        """
        start = time.time()
        print('Loading embeddings...', end=' ', flush=True)

        # allow user to provide select args in function call
        if load_all is not None:
            self.config.load_all_embeddings = load_all
        if filepath is not None:
            filepath = generic_utils.clean_path(filepath)
            self.config.pretrained_embeddings = filepath

        if not self.datasets:
            err_msg = "You need to call `load_dataset()` before calling `load_embeddings()`"
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)
        if not self.config.pretrained_embeddings:
            err_msg = ("`Saber.load_embeddings()` was called but `pretrained_embeddings` argument "
                       "is empty")
            LOGGER.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        # load the embeddings
        self.embeddings = Embeddings(filepath=self.config.pretrained_embeddings,
                                     token_map=self.datasets[0].type_to_idx['word'],
                                     debug=self.config.debug)
        # when all embeddings loaded a new type_to_index mapping is generated, so update datasets
        # current type to index mappings
        if self.config.load_all_embeddings:
            type_to_idx = self.embeddings.load(binary, load_all=self.config.load_all_embeddings)
            for dataset in self.datasets:
                word_types = list(dataset.type_to_idx['word'])
                char_types = list(dataset.type_to_idx['char'])
                dataset.type_to_idx['word'] = Preprocessor.type_to_idx(word_types, type_to_idx['word'])
                dataset.type_to_idx['char'] = Preprocessor.type_to_idx(char_types, type_to_idx['char'])
                dataset.get_idx_seq()
        else:
            self.embeddings.load(binary)

        end = time.time() - start
        embed, found, dims = self.embeddings.num_embed, self.embeddings.num_found, self.embeddings.dimension
        info_msg = 'Loaded {}/{} word vectors of dimension {}.'.format(embed, found, dims)
        print('Done ({0:.2f} seconds).'.format(end))
        print(info_msg)
        LOGGER.info(info_msg)

    def build(self, model_name=None):
        """Specifies and compiles the chosen sequence model, given by 'self.config.model_name'.

        For a chosen sequence model class (provided at the command line or in the configuration file
        and saved as 'self.config.model_name'), "specify"s and "compile"s the Keras model(s) it
        contains.

        Raises:
            ValueError: If `self.datasets` is None or `self.config.model_name` is not valid.
        """
        start_time = time.time()

        if not self.datasets:
            err_msg = "You need to call 'Saber.load_dataset()' before calling 'Saber.build()'"
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        if model_name is not None:
            self.config.model_name = model_name.lower().strip()
        if self.config.model_name not in constants.MODEL_NAMES:
            err_msg = "`model_name` must be one of: {}, got {}".format(constants.MODEL_NAMES,
                                                                       self.config.model_name)
            LOGGER.error('ValueError: %s ', err_msg)
            raise ValueError(err_msg)

        # setup the chosen model
        if self.config.model_name == 'mt-lstm-crf':
            print('Building the multi-task BiLSTM-CRF model...', end=' ', flush=True)
            from .models.multi_task_lstm_crf import MultiTaskLSTMCRF
            model = MultiTaskLSTMCRF(config=self.config,
                                     datasets=self.datasets,
                                     embeddings=self.embeddings)

        # create the model
        model.specify()
        model.compile()
        self.model = model

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds).'.format(elapsed_time))
        LOGGER.info('%s model was built successfully', self.config.model_name.upper())

        if self.config.verbose:
            for i, model in enumerate(self.model.models):
                ds_name = os.path.basename(self.config.dataset_folder[i])
                print('Model architecture for dataset {}:'.format(ds_name))
                model.summary()

    def train(self):
        """Initiates training of model at `self.model`.

        Raises:
            MissingStepException: If `self.datasets` or `self.model` are None.
        """
        if not self.datasets:
            err_msg = "You need to call `Saber.load_dataset()` before calling `Saber.train()`"
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)
        if not self.model:
            err_msg = "You need to call `Saber.build()` before calling `Saber.train()`"
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        trainer = Trainer(self.config, self.datasets, self.model)
        trainer.train()

# https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class MissingStepException(Exception):
    """Execption subclass for signalling to user that some required previous step was missed."""
    pass
