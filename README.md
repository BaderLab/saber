[![Coverage Status](https://coveralls.io/repos/github/BaderLab/multi-task-learning-BNER-trigger-word/badge.svg?branch=master)](https://coveralls.io/github/BaderLab/multi-task-learning-BNER-trigger-word?branch=master)
[![Build Status](https://travis-ci.org/BaderLab/multi-task-learning-BNER-trigger-word.svg?branch=master)](https://travis-ci.org/BaderLab/multi-task-learning-BNER-trigger-word)

# Kari

**Kari** is a deep-learning based tool for **information extraction** in the biomedical domain.

### Requirements

Requires `Python >= 3.4`. Requirements can be installed by calling `pip install -r requirements.txt`.

### Usage

All hyper-parameters are specified in a configuration file. The configuration file can be specified when running __Kari__

```
python main.py --config_filepath path/to/config.ini
```

If not specified, the default config file (`kari/config.ini`) is loaded.
