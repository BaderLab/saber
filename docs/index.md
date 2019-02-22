__Saber__ (**S**equence **A**nnotator for **B**iomedical **E**ntities and **R**elations) is a deep-learning based tool for __information extraction__ in the biomedical domain.

The neural network model used is a BiLSTM-CRF [[1](https://arxiv.org/abs/1603.01360), [2](https://arxiv.org/abs/1603.01354)]; a state-of-the-art architecture for sequence labelling. The model is implemented using [Keras](https://keras.io/) / [Tensorflow](https://www.tensorflow.org).

The goal is that Saber will eventually perform all the important steps in text-mining of biomedical literature:

- Coreference resolution (:white_check_mark:)
- Biomedical named entity recognition (BioNER) (:white_check_mark:)
- Entity linking / grounding / normalization (:white_check_mark:)
- Simple relation extraction (:soon:)
- Event extraction (:soon:)

Pull requests are welcome! If you encounter any bugs, please open an issue in the [GitHub repository](https://github.com/BaderLab/saber).
