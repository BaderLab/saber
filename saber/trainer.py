"""Contains the Trainer class, which coordinates the training of Keras ML models for Saber.
"""
import logging
import random

from sklearn.model_selection import train_test_split

from .utils import model_utils

class Trainer(object):
    """A class for co-ordinating the training of Keras model(s).

    Args:
        config (Config): instance of Config which contains all model and hyperparameter
            specifications.
        ds (list): a list containing one or more Dataset objects
        model (BaseModel): an instance of BaseModel
    """
    def __init__(self, config, ds, model):
        # config contains a dictionary of hyperparameters
        self.config = config
        # dataset(s) tied to this instance
        self.ds = ds
        # model tied to this instance
        self.model = model
        # metric(s) object tied to this instance, one per dataset
        # self.metrics = []
        self.log = logging.getLogger(__name__)

    def _specify(self):
        """Calls self.model.specify_() function, which specifies the models architecture."""
        self.model.specify_()

    def _compile(self):
        """Calls self.model.compile_() function, which compiles the model."""
        self.model.compile_()

    def _clear_model(self):
        """Clears the model tied to Trainer instance."""
        self.model.model = []

    def train(self, callbacks, output_dir):
        """Co-ordinates the training of Keras model(s) at self.model.model.

        For a given list of Keras models (self.model.model), collects all training data from self.ds
        (where the data for the i-th model in self.model.model is stored at ds[i]), and determines
        whether to train using k-fold cross-validation (CV) or a simple train/valid/test strategy.
        CV is used when no valid or test partitions are provided, and a train/valid/test strategy is
        used otherwise.

        Args:
            callbacks (dict): dictionary contaiing Keras callback objects.
            output_dir (str): list of filepaths to save model output to, one for each model.
        """
        # collect training data (train and optionally valid/test) for all ds
        training_data = {}
        for i, ds in enumerate(self.ds):
            training_data[i] = self._collect_data(ds)
        # TODO: ugly, is there a better way to check for this?
        # TODO: what if diff ds follow diff schemes?
        # no valid or test sets provided, perform cross-validation
        if training_data[0]['X_test'] is None and training_data[0]['X_valid'] is None:
            self._cross_val(training_data, output_dir, callbacks)
        # valid/test set was provided, train via standard train/valid/test split
        else:
            self._train_valid_test(training_data, output_dir, callbacks)

    def _collect_data(self, ds):
        """Collect data to use during training of models.

        Collects the data to be used during training of models. Expects dataset 'ds' to contain a
        train partition (ds.idx_seq['train']). Additionally loads valid and test partitions
        (ds.idx_seq['valid'] and (ds.idx_seq['test']) if 'valid' and 'test' exist in
        ds.partition_filepaths. (ds.idx_seq['valid']

        Args:
            ds (Dataset): dataset object

        Returns:
            a dictionary containing data and lables for a train partition, and optionally, valid and
            test partitions where keys 'X_<partition>' and 'y_<partiton>' point to the data and
            labels for partitions 'train', 'valid', or 'test' respectively.
        """
        # minimally, collect data for training partition
        training_data = {
            'X_train': [ds.idx_seq['train']['word'], ds.idx_seq['train']['char']],
            'X_valid': None,
            'X_test': None,
            'y_train': ds.idx_seq['train']['tag'],
            'y_valid': None,
            'y_test': None,
        }

        # collect valid partition data (if it exists)
        if 'valid' in ds.partition_filepaths:
            valid_data = {
                'X_valid': [ds.idx_seq['valid']['word'], ds.idx_seq['valid']['char']],
                'y_valid': ds.idx_seq['valid']['tag']
            }
            training_data.update(valid_data)

        # collect test partition data (if it exists)
        if 'test' in ds.partition_filepaths:
            test_data = {
                'X_test': [ds.idx_seq['test']['word'], ds.idx_seq['test']['char']],
                'y_test': ds.idx_seq['test']['tag']
            }
            training_data.update(test_data)

        return training_data

    def _split_train_valid(self, data, train_size=0.90):
        """Splits training data into train/valid partitions.

        For training data 'data', splits the training partition into two new partitions: a new
        training partition of size 'train_size' percent of the original training partition, and a
        valid partition of size 1 - 'train_size' percent the size of the original train partition.

        Args:
            data (dict): a dictionary containing the training data
            train_size (float): percent of training partiton to retain, the reminader
                (1 - train_size) is used as validation data.
        Returns:
            data, where a new valid partition of 1 - 'train_size' has been added
            ('X_valid, 'y_valid')

        Preconditions:
            expects data to be a dictionary containing keys 'X_train', which indexes into a list,
            which contains word sequences as its first item (numpy array of shape (num sentences,
            num words)) and character sequences (num sentences, num words, num characters) as its
            second, and, 'y_train' which indexes into a tag sequence (numpy array of shape
            (num sentence, num words, num tags))

        Raises:
            ValueError, if data does not contain the keys 'X_train' and 'y_train'
        """
        valid_size = 1 - train_size
        try:
            X_train_word, X_valid_word, X_train_char, X_valid_char, y_train, \
                y_valid = train_test_split(
                    data['X_train'][0],
                    data['X_train'][1],
                    data['y_train'],
                    train_size=train_size,
                    test_size=valid_size,
                    random_state=42,
                    shuffle=False)
        except KeyError:
            self.log.error(("ValueError raised because 'data' argument passed to Trainer."
                            "_split_train_valid() did not contain the keys 'X_train' and "
                            "'y_train'"))
            raise ValueError("'data' must contain the keys 'X_train' and 'y_train'")

        data['X_train'] = [X_train_word, X_train_char]
        data['X_valid'] = [X_valid_word, X_valid_char]
        data['y_train'] = y_train
        data['y_valid'] = y_valid

        return data

    def _train_valid_test(self, training_data, output_dir, callbacks):
        """Trains a Keras model with a standard train/valid/test strategy.

        Trains a Keras model (self.model.model) or models in the case of multi-task learning
        (self.model.model is a list of Keras models) using a simple train/valid/test partition.
        Minimally expects a train partition and one or both of valid and test partitions to be
        supplied in 'training_data'.

        Args:
            training_data (dict): a dictionary of dictionaries, where the first set of keys are
                dataset indices (0, 1, ...) and the second set of keys are dataset partitions
                ('X_train', 'y_train', 'X_valid', ...)
            output_dir (lst): a list of output directories, one for each dataset
            callbacks: a Keras CallBack object for per epoch model checkpointing.
        """
        print('Using train/test/valid strategy...')
        self.log.info('Using a train/test/valid strategy for training')
        # if validation data is provided, use it, otherwise take 10% of
        # train set as validation data
        for i, ds in enumerate(self.ds):
            if training_data[i]['X_valid'] is None:
                print(('No validation set was provided, using 10% of training set selected '
                       'at random as the validation data'))
                training_data[i] = self._split_train_valid(training_data[i])

        # get list of Keras Callback objects for computing/storing metrics
        metrics = model_utils.get_metrics(datasets=self.ds,
                                          training_data=training_data,
                                          output_dir=output_dir,
                                          criteria=self.config.criteria)

        # Epochs
        for epoch in range(self.config.epochs):
            print('Global epoch: {}/{}'.format(epoch + 1, self.config.epochs))
            # Dataset / Models
            # get a random ordering of the dataset/model indices
            ds_idx = random.sample(range(0, len(self.ds)), len(self.ds))
            for i in ds_idx:
                self.model.model[i].fit(
                    x=training_data[i]['X_train'],
                    y=training_data[i]['y_train'],
                    batch_size=self.config.batch_size,
                    epochs=1,
                    callbacks=[v[i] for k, v in callbacks.items() if v is not None] + [metrics[i]],
                    validation_data=(training_data[i]['X_valid'], training_data[i]['y_valid']),
                    verbose=1)

    def _cross_val(self, training_data, output_dir, callbacks):
        """Trains a Keras model with a cross-validation strategy.

        Trains a Keras model (self.model.model) or models in the case of multi-task learning
        (self.model.model is a list of Keras models) using a cross-validation strategy. Expects only
        a train partition to be supplied in 'training_data'.

        Args:
            training_data (dict): a dictionary of dictionaries, where the first set of keys are
                dataset indices (0, 1, ...) and the second set of keys are dataset partitions
                ('X_train', 'y_train', 'X_valid', ...)
            output_dir (lst): a list of output directories, one for each dataset
            callbacks: a Keras CallBack object for per epoch model checkpointing.
        """
        print('Using {}-fold cross-validation strategy...'.format(self.config.k_folds))
        self.log.info('Using a %s-fold cross-validation strategy for training', self.config.k_folds)
        # get train/valid indicies for each fold and dataset
        train_valid_indices = \
            model_utils.get_train_valid_indices(training_data=training_data,
                                                k_folds=self.config.k_folds)

        # Folds
        for fold in range(self.config.k_folds):

            # get the train/valid partitioned data for all datasets
            partitioned_data = model_utils.get_data_partitions(training_data, \
                train_valid_indices, fold)

            # get list of Keras Callback objects for computing/storing metrics
            metrics = model_utils.get_metrics(datasets=self.ds,
                                              training_data=partitioned_data,
                                              output_dir=output_dir,
                                              criteria=self.config.criteria,
                                              fold=fold)

            # Epochs
            for epoch in range(self.config.epochs):
                print('Fold: {}/{}; Global epoch: {}/{}'.format(fold + 1,
                                                                self.config.k_folds,
                                                                epoch + 1,
                                                                self.config.epochs))

                # Datasets / Models
                # get a random ordering of the dataset/model indices
                ds_idx = random.sample(range(0, len(self.ds)), len(self.ds))
                for i in ds_idx:

                    self.model.model[i].fit(
                        x=partitioned_data[i]['X_train'],
                        y=partitioned_data[i]['y_train'],
                        batch_size=self.config.batch_size,
                        epochs=1,
                        callbacks=[v[i] for k, v in callbacks.items() if v is not None] + \
                            [metrics[i]],
                        validation_data=(partitioned_data[i]['X_valid'],
                                         partitioned_data[i]['y_valid']),
                        verbose=1)

            # self.metrics.append(metrics_current_fold)

            # destroys the current TF graph and creates a new one, useful to
            # avoid clutter from old models / layers.
            # K.clear_session()

            # End of a k-fold, so clear the model, specify and compile again.
            # Do not clear the last model.
            if fold < self.config.k_folds - 1:
                self._clear_model()
                self._specify()
                self._compile()
