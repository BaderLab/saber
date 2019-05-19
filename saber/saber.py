"""Contains the Saber class, which exposes all of Sabers main functionality.
"""
import glob
import logging
import os
import pickle
import time
from itertools import chain
from operator import itemgetter
from pprint import pprint

import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
from keras.models import Model
from spacy import displacy

from . import constants
from .config import Config
from .dataset import Dataset
from .embeddings import Embeddings
from .preprocessor import Preprocessor
from .utils import data_utils, generic_utils, grounding_utils, model_utils

print('Saber version: {0}'.format(constants.__version__))
LOGGER = logging.getLogger(__name__)


class Saber():
    """The interface for Saber, exposing all of its main functionality.

    As the interface to Saber, this class exposes all of Sabers main functionality, including the
    training, saving, and loading of models for natural language processing of biomedical text.

    Attributes:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line. If not provided, a new instance of
            Config with default values is used (this is fine for most use cases).
    """
    def __init__(self, config=None, **kwargs):
        self.config = Config() if config is None else config

        self.preprocessor = None  # object for text processing
        self.datasets = []  # dataset(s) tied to this instance
        self.embeddings = None  # pre-trained token embeddings tied to this instance
        self.models = []  # model(s) object tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.config.verbose:
            print('Hyperparameters and model details:')
            pprint({arg: getattr(self.config, arg) for arg in constants.CONFIG_ARGS})

    def annotate(self, text, title='', jupyter=False, coref=False, ground=False):
        """Uses trained model(s) (`self.models`) to annotate `text`.

        For the model(s) at `self.models`, annotates `text`. Returns a dictionary containing the
        cleaned text ('text'), and any annotations made by the model ('ents'). If `jupyter` is True,
        renders a HTML visualization of the annotations made by the model, for use in a jupyter
        notebook.

        text (str): Raw text to annotate.
        title (str): Optional, title of the document. Defaults to the empty string ''.
        jupyter (bool): Optional, True if annotations made by the model should be rendered in HTML,
            which can be visualized in a jupter notebook. Defaults to False.
        coref (bool): Optional, True if coreference resolution should be performed before
            annotation. Defaults to False.
        ground (bool): Optional, True if annotations made by the model should be grounded to unique
            identifiers in external knowledge bases and ontologies.

        Returns:
            Dictionary containing the processed input text (`text`) and any annotations made by the
            model (`ents`).

        Raises:
            ValueError: If `text` is invalid (not a string, or empty/falsey).
        """
        if self.preprocessor is None:
            start = time.time()
            print('Loading preprocessor...', end=' ', flush=True)
            self.preprocessor = Preprocessor()
            print(f'Done ({time.time() - start:.2f} seconds).')

        if not isinstance(text, str) or not text:
            err_msg = f'Expected non-empty string for argument `text`. Got: {text}'
            LOGGER.error("ValueError: %s", err_msg)
            raise ValueError(err_msg)

        # Process raw text, flatten char offsets
        processed_text, sents, char_offsets = self.preprocessor.transform(text, coref=coref)
        char_offsets = list(chain.from_iterable(char_offsets))

        annotation = {'title': title, 'text': processed_text, 'ents': [],
                      # TODO (John): This is here strictly to prevent an error on SpaCy v2.1
                      'settings': {}}

        for model in self.models:
            X, y_preds = model.predict(sents)
            X = X.ravel()

            if not isinstance(y_preds, list):
                y_preds = [y_preds]

            for y_pred, dataset in zip(y_preds, model.datasets):
                y_pred = np.argmax(y_pred, axis=-1).ravel()

                # mask predictions where the input to the model was a pad
                X, y_pred = \
                    model_utils.mask_labels(X, y_pred, dataset.type_to_idx['word'][constants.PAD])

                # TODO (John): This should be handeled in `model.predict`. See my QA model for help.
                # remove wordpiece tags (bert-specific)
                pred_tag_seq = [dataset.idx_to_tag[idx] for idx in y_pred
                                if dataset.idx_to_tag[idx] != constants.WORDPIECE]

                # Accumulate predicted entities
                for chunk in self.preprocessor.chunk_entities(pred_tag_seq):
                    # Chunks are tuples (label, start, end), offsets is a lists of lists of tuples
                    start = char_offsets[chunk[1]][0]
                    end = char_offsets[chunk[-1] - 1][-1]
                    text = processed_text[start:end]
                    label = chunk[0]

                    annotation['ents'].append({'start': start,
                                               'end': end,
                                               'text': text,
                                               'label': label})

        # Need to sort by first apperance or displacy will duplicate text
        annotation['ents'] = sorted(annotation['ents'], key=itemgetter('start'))

        if ground:
            try:
                annotation = grounding_utils.ground(annotation)
            except Exception as e:
                err_msg = ('Grounding step in `Saber.annotate()` failed'
                           f' (`grounding_utils.ground()` threw error: {e}. Check that you'
                           ' have a stable internet connection.')
                LOGGER.error(err_msg)

        if jupyter:
            displacy.render(annotation, jupyter=True, style='ent', manual=True,
                            options=constants.OPTIONS)

        return annotation

    def save(self, directory=None, compress=True, model_idx=-1, output_indices=None):
        """Saves the Saber model at `self.models[model_idx]` to disk.

        Saves the necessary files for model persistence to `directory`. If not provided, `directory`
        defaults to '<self.config.output_folder>/<constants.PRETRAINED_MODEL_DIR>/<dataset_names>'.
        Which model to be saved to disk can be selected by providing an index into `self.models`
        with `model_idx`, and which output layers from this model that should be retained (in the
        case of a multi-task model) can be provided as a list of indices with `output_indices`. By
        default, the last loaded model (`self.models[-1]`) with all of its output layers is saved to
        disk.

        Args:
            directory (str): Optional, directory path to save model to. If None, a default directory
                under `self.config.output_folder` is created.
            compress (bool): Optional, True if model should be saved as tarball. Defaults to True.
            model_idx (int): Optional, an index into `self.models` corresponding to which model to
                save in the case of multiple loaded models. Defaults to -1, which will save the last
                loaded model (`self.models[-1]`) to disk.
            output_indices (list): Optional, a list of indexes representing which output layers from
                `self.models[-1]` to save in the case of a multi-task model. Defaults to None, which
                will include all classifiers under the saved model.
        """
        start = time.time()

        if not self.models:
            err_msg = ('`Saber.models` is empty. Make sure to load at least one model with'
                       ' `Saber.load()` or `Saber.build()` before calling `Saber.save()`.')
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        saber_model = self.models[model_idx]
        print(f'Saving model {saber_model}...', end=' ', flush=True)

        type_to_idx, idx_to_tag = [], []

        # Allows user to call save without providing a directory, default one is chosen
        if directory is None:
            directory = model_utils.prepare_pretrained_model_dir(self.config)
        directory = generic_utils.clean_path(directory)
        generic_utils.make_dir(directory)

        # retain only output layers specified in output_indices
        if output_indices:
            # Ignore if this is not a MTM
            if saber_model.framework == constants.KERAS and isinstance(saber_model.model.output, list):
                last_hidden_layer = saber_model.model.layers[-(len(saber_model.model.output) + 1)]
                outputs = [saber_model.model.ouput[i](last_hidden_layer) for i in output_indices]

                model_to_save = Model(saber_model.model.input, outputs)
            elif saber_model.framework == constants.PYTORCH:
                # TODO (John): Will write this when MultiTaskBert is finished
                # classifier_indices = ... if classifier_indices is None else classifier_indices
                pass

            for i, dataset in enumerate(saber_model.datasets):
                if i in output_indices:
                    type_to_idx.append(dataset.type_to_idx)
                    idx_to_tag.append(dataset.idx_to_tag)
        else:
            model_to_save = saber_model.model

            for _, dataset in enumerate(saber_model.datasets):
                type_to_idx.append(dataset.type_to_idx)
                idx_to_tag.append(dataset.idx_to_tag)

        # save attributes that we need for inference
        attributes_filepath = os.path.join(directory, constants.ATTRIBUTES_FILENAME)

        model_attributes = {'model_name': saber_model.model_name,
                            'framework': saber_model.framework,
                            'type_to_idx': type_to_idx,
                            'idx_to_tag': idx_to_tag,
                            }

        # saving Keras models requires a seperate file for weights
        if saber_model.framework == constants.KERAS:
            model_filepath = os.path.join(directory, constants.KERAS_MODEL_FILENAME)
            weights_filepath = os.path.join(directory, constants.WEIGHTS_FILENAME)
            saber_model.save(model_filepath, weights_filepath, model=model_to_save)
        elif saber_model.framework == constants.PYTORCH:
            if saber_model.model_name == 'bert-ner':
                model_attributes['pretrained_model_name_or_path'] = \
                    saber_model.pretrained_model_name_or_path
            model_filepath = os.path.join(directory, constants.PYTORCH_MODEL_FILENAME)
            saber_model.save(model_filepath, model=model_to_save)

        pickle.dump(model_attributes, open(attributes_filepath, 'wb'))

        # save config for reproducibility
        self.config.save(directory)

        if compress:
            generic_utils.compress_directory(directory)

        print(f'Done ({time.time() - start:.2f} seconds).')
        print(f'Model was saved to {directory}')
        LOGGER.info('Model was saved to %s', directory)

        return directory

    def load(self, directory):
        """Loads the Saber model at `directory`.

        Args:
            directory (str): Directory path to the saved model. Can be a path on disk, or one of
                `constants.PRETRAINED_MODELS`, in which case a model is retrived from Google Drive.
        """
        start = time.time()
        print(('Loading model(s):'
               f' {directory if not isinstance(directory, list) else ", ".join(directory)}...'),
              end=' ', flush=True)

        if not isinstance(directory, list):
            directory = [directory]

        for dir_ in directory:
            # get what might be a pretrained model name
            pretrained_model = os.path.splitext(dir_)[0].strip().upper()

            # allows user to provide names of pre-trained models (e.g. 'PRGE-base')
            if pretrained_model in constants.PRETRAINED_MODELS:
                dir_ = os.path.join(constants.PRETRAINED_MODEL_DIR, pretrained_model)
                # download model from Google Drive, will skip if already exists
                file_id = constants.PRETRAINED_MODELS[pretrained_model]
                dest_path = '{}.tar.bz2'.format(dir_)
                gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path)

                LOGGER.info('Loaded pre-trained model %s from Google Drive', pretrained_model)

            dir_ = generic_utils.clean_path(dir_)
            generic_utils.extract_directory(dir_)

            # load only the objects we need for model prediction
            attributes_filepath = os.path.join(dir_, constants.ATTRIBUTES_FILENAME)
            model_attributes = pickle.load(open(attributes_filepath, "rb"))

            # its easiest to use a Dataset object to hold these data objects
            datasets = []
            attributes_for_inference = zip(model_attributes['type_to_idx'],
                                           model_attributes['idx_to_tag'])
            for type_to_idx, idx_to_tag in attributes_for_inference:
                dataset = Dataset()
                dataset.type_to_idx = type_to_idx
                dataset.idx_to_tag = idx_to_tag

                datasets.append(dataset)

            # prevents user from having to specify pre-trained models name
            self.config.model_name = model_attributes['model_name']

            # keras models are saved as two files (model.* and weights.*)
            if model_attributes['framework'] == constants.KERAS:
                weights_filepath = os.path.join(dir_, constants.WEIGHTS_FILENAME)
                pretrained_model_name_or_path = None
            # we need to know what pre-trained BERT model was used to load the correct weights
            elif model_attributes['model_name'] == 'bert-ner':
                weights_filepath = None
                pretrained_model_name_or_path = \
                    model_attributes.get('pretrained_model_name_or_path')

            model_filepath = glob.glob(os.path.join(dir_, 'model.*'))[0]

            saber_model = model_utils.load_pretrained_model(
                config=self.config,
                datasets=datasets,
                model_filepath=model_filepath,
                weights_filepath=weights_filepath,
                pretrained_model_name_or_path=pretrained_model_name_or_path
            )

            self.models.append(saber_model)

        print(f'Done ({time.time() - start:.2f} seconds).')

        return directory

    def load_dataset(self, directory=None):
        """Loads a Saber dataset at `directory`.

        Args:
            directory (str or list): Optional, path to a dataset folder(s). If not None, overwrites
                `self.config.dataset_folder`

        Raises:
            ValueError: If `self.config.dataset_folder` is None and `directory` is None.
        """
        start = time.time()

        # allows a user to provide the dataset directory when function is called
        if directory is not None:
            dataset_folder = directory if isinstance(directory, list) else [directory]
            dataset_folder = [generic_utils.clean_path(dir_) for dir_ in dataset_folder]
            self.config.dataset_folder = dataset_folder

        if not self.config.dataset_folder:
            err_msg = ('Expected a valid path in "directory" or "self.config.dataset_folder".'
                       f' Got {directory} and {self.config.dataset_folder} respectively')
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        # datasets may be 'single' or 'compound' (more than one)
        if len(self.config.dataset_folder) == 1:
            print('Loading (single) dataset...', end=' ', flush=True)
            self.datasets = data_utils.load_single_dataset(self.config)
            LOGGER.info('Loaded single dataset at: %s', self.config.dataset_folder)
        else:
            print('Loading (compound) dataset...', end=' ', flush=True)
            self.datasets = data_utils.load_compound_dataset(self.config)
            LOGGER.info('Loaded multiple datasets at: %s', self.config.dataset_folder)

        # TODO: This whole block assumes we are transfering from last model loaded.
        # Might need a better scheme.
        # if a model has already been loaded, assume we are transfer learning
        if self.models:
            print('\nTransferring from pre-trained model...', end=' ', flush=True)
            LOGGER.info('Transferred from a pre-trained model')

            type_to_idx = self.models[-1].datasets[0].type_to_idx

            # make required changes to target datasets
            for dataset in self.datasets:
                data_utils.setup_dataset_for_transfer(dataset, type_to_idx)
            # prepare model for transfer learning
            self.models[-1].prepare_for_transfer(self.datasets)

        print(f'Done ({time.time() - start:.2f} seconds).')

    def load_embeddings(self, filepath=None, binary=True, load_all=None):
        """Loads pre-trained token embeddings at `filepath`.

        Args:
            filepath (str): Optional, Path to pre-trained embeddings file. If not None, overwrites
                'self.config.pretrained_embeddings'. Defaults to None.
            binary (bool): Optional, True if pre-trained embeddings are in C binary format, False if
                they are in C text format.
            load_all (bool): Optional, True if all pre-trained embeddings should be loaded.
                Otherwise only embeddings for tokens found in the training set are loaded. Defaults
                to None.

        Raises:
            MissingStepException: If no dataset has been loaded (`self.datasets` is `None`).
            ValueError: If 'self.config.pretrained_embeddings' is `None` and `filepath` is `None`.
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
            err_msg = ('`Saber.datasets` is empty. Make sure to load at least one dataset with'
                       ' `Saber.load_dataset()` before calling `Saber.load_embeddings()`.')
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        if not self.config.pretrained_embeddings:
            err_msg = ('Expected a valid path in "filepath" or "self.config.pretrained_embeddings".'
                       f' Got {filepath} and {self.config.pretrained_embeddings} respectively')
            LOGGER.error('ValueError %s', err_msg)
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
                dataset.type_to_idx['word'] = \
                    Preprocessor.type_to_idx(word_types, type_to_idx['word'])
                dataset.type_to_idx['char'] = \
                    Preprocessor.type_to_idx(char_types, type_to_idx['char'])
                dataset.get_idx_seq()
        else:
            self.embeddings.load(binary)

        embed = self.embeddings.num_embed
        found = self.embeddings.num_found
        dims = self.embeddings.dimension

        info_msg = f'Loaded {embed}/{found} word vectors of dimension {dims}.'
        print(info_msg)
        print(f'Done ({time.time() - start:.2f} seconds).')

        LOGGER.info(info_msg)

    def build(self, model_name=None):
        """Specifies and compiles the chosen sequence model, given by 'self.config.model_name'.

        For a chosen sequence model class (provided at the command line or in the configuration file
        and saved as 'self.config.model_name'), "specify"s and "compile"s the Keras model(s) it
        contains.

        Args:
            model_name (str): Optional, one of `constants.MODEL_NAMES`. If None, equal to
                `self.config.model_name`.

        Raises:
            ValueError: If `self.datasets` is None or `self.config.model_name` is not valid.
        """
        start = time.time()

        if not self.datasets:
            err_msg = ('`Saber.datasets` is empty. Make sure to load at least one dataset with'
                       ' `Saber.load_dataset()` before calling `Saber.build()`.')
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        if model_name is not None:
            self.config.model_name = model_name.lower().strip()

        # setup the chosen model
        if self.config.model_name == 'bilstm-crf-ner':
            print('Building the BiLSTM-CRF model for NER...', end=' ', flush=True)
            from .models.multi_task_lstm_crf import MultiTaskLSTMCRF
            saber_model = MultiTaskLSTMCRF(config=self.config, datasets=self.datasets,
                                           embeddings=self.embeddings)
        elif self.config.model_name == 'bert-ner':
            print('Building the BERT model for NER...', end=' ', flush=True)
            from .models.bert_token_classifier import BertTokenClassifier
            saber_model = \
                BertTokenClassifier(config=self.config, datasets=self.datasets,
                                    pretrained_model_name_or_path=constants.PYTORCH_BERT_MODEL)
        # TODO: This should be handled in config, by supplying a list of options.
        else:
            err_msg = (f'`model_name` must be one of: {constants.MODEL_NAMES},'
                       ' got {self.config.model_name}')
            LOGGER.error('ValueError: %s ', err_msg)
            raise ValueError(err_msg)

        saber_model.specify()
        saber_model.compile()  # does nothing if not a Keras model

        self.models.append(saber_model)

        print(f'Done ({time.time() - start:.2f} seconds).')
        LOGGER.info('%s model was built successfully', self.config.model_name.upper())

        if self.config.verbose:
            print('Model architecture:')
            # TODO (John) Not implemented for PyTorch models, will print nothing.
            saber_model.summary()

    def train(self, model_idx=-1):
        """Initiates training of model at `self.model[model_idx]`.

        Args:
            model_idx (int): Optional, an index into `self.models` corresponding to which model to
                train in the case of multiple loaded models. Defaults to -1, which will train the
                last loaded model (`self.models[-1]`).

        Raises:
            MissingStepException: If `self.datasets` or `self.model` are None.
        """
        if not self.models:
            err_msg = ('`Saber.models` is empty. Make sure to load at least one model with'
                       ' `Saber.load()` or `Saber.build()` before calling `Saber.train()`.')
            LOGGER.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)

        self.models[model_idx].train()


# https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class MissingStepException(Exception):
    """Exception subclass for signalling to user that some required previous step was missed."""
    pass
