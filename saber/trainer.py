"""Contains the Trainer class, which coordinates the training of Keras ML models for Saber.
"""
import logging
import random

from .utils import data_utils, model_utils

LOGGER = logging.getLogger(__name__)

class Trainer(object):
    """A class for co-ordinating the training of Keras model(s).

    Args:
        config (Config): A Config object which contains a set of harmonized arguments provided in
            a *.ini file and, optionally, from the command line.
        datasets (list): A list containing one or more Dataset objects.
        model (BaseModel): The model to train.
    """
    def __init__(self, config, datasets, model):
        self.config = config # hyperparameters and model details
        self.datasets = datasets # dataset(s) tied to this instance
        self.model = model # model tied to this instance

        self.output_dir = model_utils.prepare_output_directory(self.config)
        self.callbacks = model_utils.setup_callbacks(self.config, self.output_dir)
        self.training_data = self.model.prepare_data_for_training()

    def train(self):
        """Co-ordinates the training of Keras model(s) at `self.model.models`.

        Coordinates the training of one or more Keras models (given at `self.model.models`). If a
        valid or test set is provided (`Dataset.directory['valid']` or `Dataset.directory['test']`
        are not None) a simple train/valid/test strategy is used. Otherwise, cross-validation is
        used.

        Args:
            callbacks (dict): Dictionary containing Keras callback objects.
            output_dir (list): List of directories to save model output to, one for each model.
        """
        # TODO: ugly, is there a better way to check for this? what if dif ds follow dif schemes?
        if (self.training_data[0]['x_valid'] is not None or
                self.training_data[0]['x_test'] is not None):
            self._train_valid_test()
        else:
            self._cross_validation()

    def _train_valid_test(self):
        """Trains a Keras model with a standard train/valid/test strategy.

        Trains a Keras model (`self.model.models`), or models in the case of multi-task learning
        (`self.model.models` is a list of Keras models) using a simple train/valid/test strategy.
        Minimally expects a train partition and one or both of valid and test partitions to be
        supplied in the Dataset objects at `self.datasets`.

        Args:
            callbacks (dict): Dictionary containing Keras callback objects.
            output_dir (list): List of directories to save model output to, one for each model.
        """
        print('Using train/test/valid strategy...')
        LOGGER.info('Using a train/test/valid strategy for training')
        # use 10% of train data as validation data if no validation data provided
        if self.training_data[0]['x_valid'] is None:
            self.training_data = data_utils.collect_valid_data(self.training_data)
        # get list of Keras Callback objects for computing/storing metrics
        metrics = model_utils.setup_metrics_callback(config=self.config,
                                                     datasets=self.datasets,
                                                     training_data=self.training_data,
                                                     output_dir=self.output_dir)
        # training loop
        for epoch in range(self.config.epochs):
            print('Global epoch: {}/{}\n{}'.format(epoch + 1, self.config.epochs, '-' * 20))
            # get a random ordering of the dataset/model indices
            ds_idx = random.sample(range(0, len(self.datasets)), len(self.datasets))
            for i in ds_idx:
                self.model.models[i].fit(x=self.training_data[i]['x_train'],
                                         y=self.training_data[i]['y_train'],
                                         batch_size=self.config.batch_size,
                                         callbacks=[cb[i] for cb in self.callbacks] + [metrics[i]],
                                         validation_data=(self.training_data[i]['x_valid'],
                                                          self.training_data[i]['y_valid']),
                                         verbose=1,
                                         # required for Keras to properly display current epoch
                                         initial_epoch=epoch,
                                         epochs=epoch + 1)

    def _cross_validation(self):
        """Trains a Keras model with a cross-validation strategy.

        Trains a Keras model (self.model.models) or models in the case of multi-task learning
        (self.model.models is a list of Keras models) using a cross-validation strategy. Expects
        only a train partition to be supplied in `training_data`.

        Args:
            training_data (dict): a dictionary of dictionaries, where the first set of keys are
                dataset indices (0, 1, ...) and the second set of keys are dataset partitions
                ('X_train', 'y_train', 'X_valid', ...)
            output_dir (lst): a list of output directories, one for each dataset
            callbacks: a Keras CallBack object for per epoch model checkpointing.
        """
        print('Using {}-fold cross-validation strategy...'.format(self.config.k_folds))
        LOGGER.info('Using a %s-fold cross-validation strategy for training', self.config.k_folds)
        # get the train/valid partitioned data for all datasets and all folds
        self.training_data = data_utils.collect_cv_data(self.training_data, self.config.k_folds)
        # training loop
        for fold in range(self.config.k_folds):
            # get list of Keras Callback objects for computing/storing metrics
            metrics = model_utils.setup_metrics_callback(config=self.config,
                                                         datasets=self.datasets,
                                                         training_data=self.training_data,
                                                         output_dir=self.output_dir,
                                                         fold=fold)
            for epoch in range(self.config.epochs):
                train_info = (fold + 1, self.config.k_folds, epoch + 1, self.config.epochs)
                print('Fold: {}/{}; Global epoch: {}/{}\n{}'.format(*train_info, '-' * 30))
                # get a random ordering of the dataset/model indices
                ds_idx = random.sample(range(0, len(self.datasets)), len(self.datasets))
                for i in ds_idx:
                    self.model.models[i].fit(
                        x=self.training_data[i][fold]['x_train'],
                        y=self.training_data[i][fold]['y_train'],
                        batch_size=self.config.batch_size,
                        callbacks=[cb[i] for cb in self.callbacks] + [metrics[i]],
                        validation_data=(self.training_data[i][fold]['x_valid'],
                                         self.training_data[i][fold]['y_valid']),
                        verbose=1,
                        # required for Keras to properly display current epoch
                        initial_epoch=epoch,
                        epochs=epoch + 1)

            # clear and rebuild the model at end of each fold (except for the last fold)
            if fold < self.config.k_folds - 1:
                self._reset_model()

    def _reset_model(self):
        """Clears and rebuilds the model at the end of a cross-validation fold.
        """
        # destroys current TF graph and creates new one, useful for avoiding clutter from old models
        # K.clear_session()
        self.model.models = []
        self.model.specify()
        self.model.compile()
