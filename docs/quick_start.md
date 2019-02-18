# Quick Start

If your goal is to use Saber to annotate biomedical text, then you can either use the [web-service](#web-service) or a [pre-trained model](#pre-trained-models). If you simply want to check Saber out, without installing anything locally, try the [Google Colaboratory](#google-colaboratory) notebook.

## Google Colaboratory

The fastest way to check out Saber is by following along with the Google Colaboratory notebook ([![Colab](https://img.shields.io/badge/launch-Google%20Colab-orange.svg?style=flat-square)](https://colab.research.google.com/drive/1WD7oruVuTo6p_908MQWXRBdLF3Vw2MPo)). In order to be able to run the cells, select "Open in Playground" or, alternatively, save a copy to your own Google Drive account (File > Save a copy in Drive).

## Web-service

To use Saber as a **local** web-service, run

```
(saber) $ python -m saber.cli.app
```

or, if you prefer, you can pull & run the Saber image from **Docker Hub**

```
# Pull Saber image from Docker Hub
$ docker pull pathwaycommons/saber
# Run docker (use `-dt` instead of `-it` to run container in background)
$ docker run -it --rm -p 5000:5000 --name saber pathwaycommons/saber
```

!!! tip
    Alternatively, you can clone the GitHub repository and build the container from the `Dockerfile` with `docker build -t saber .`

There are currently two endpoints, `/annotate/text` and `/annotate/pmid`. Both expect a `POST` request with a JSON payload, e.g.

```json
{
  "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."
}
```

or

```json
{
  "pmid": 11835401
}
```

For example, with the web-service running locally

``` bash tab="Bash"
curl -X POST 'http://localhost:5000/annotate/text' \
--data '{"text": 'The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.'}'
```

``` python tab="python"
import requests # assuming you have requests package installed!

url = "http://localhost:5000/annotate/pmid"
payload = {"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}
response = requests.post(url, json=payload)

print(response.text)
print(response.status_code, response.reason)
```

!!! warning
    The first request to the web-service will be slow (~60s). This is because a large language
    model needs to be loaded into memory.

Documentation for the Saber web-service API can be found [here](https://baderlab.github.io/saber-api-docs/). We hope to provide a live version of the web-service soon!

## Pre-trained models

First, import `Saber`. This class coordinates training, annotation, saving and loading of models and datasets. In short, this is the interface to Saber.

```python
from saber.saber import Saber
```

To load a pre-trained model, first create a `Saber` object

```python
saber = Saber()
```

and then load the model of our choice

```python
saber.load('PRGE')
```

!!! tip
    See [Resources: Pre-trained models](../resources#pre-trained-models) for pre-trained model names and details. You will need an internet connection to download a pre-trained model.

To annotate text with the model, just call the `Saber.annotate()` method

```python
saber.annotate("The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.")
```

!!! warning
    The `Saber.annotate()` method will be slow the first time you call it (~60s). This is because a large language model needs to be loaded into memory.

### Coreference Resolution

[**Coreference**](https://en.wikipedia.org/wiki/Coreference) occurs when two or more expressions in a text refer to the same person or thing, that is, they have the same **referent**. Take the following example:

_"__IL-6__ supports tumour growth and metastasising in terminal patients, and __it__ significantly engages in cancer cachexia (including anorexia) and depression associated with malignancy."_

Clearly, "__it__" referes to "__IL-6__". If we do not resolve this coreference, then "__it__" will not be labeled as an entity and any relation or event it is mentioned in will not be extracted. Saber uses [NeuralCoref](https://github.com/huggingface/neuralcoref), a state-of-the-art coreference resolution tool based on neural nets and built on top of [Spacy](https://spacy.io). To use it, just supply the argument `coref=True` (which is `False` by default) to the `Saber.annotate()` method

```python
text = "IL-6 supports tumour growth and metastasising in terminal patients, and it significantly engages in cancer cachexia (including anorexia) and depression associated with malignancy."
# WITHOUT coreference resolution
saber.annotate(text, coref=False)
# WITH coreference resolution
saber.annotate(text, coref=True)
```

!!! note
    If you are using the web-service, simply supply `"coref": true` in your `JSON` payload to resolve coreferences.

Saber currently takes the simplest possible approach: replace all coreference mentions with their referent, and then feed the resolved text to the model that identifies named entities.

### Grounding

**Grounding** (sometimes called **entity linking** or **normalization**) involves mapping each annotated entity to a unique identifier in an external resource such as a database or ontology. To ground entities in a call to `Saber.annotate()`, simply pass the argument `ground=True`

```python
saber.annotate('The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.', ground=True)
```

The grounding functionality is implemented by the [EXTRACT 2.0 API](https://extract.jensenlab.org/). Note that you will need an internet connection or grounding will fail. Also note that `Saber.annotate()` will take slightly longer to return a response when `ground=True` (up to a few seconds).

See [Resources: Pre-trained models](../resources#pre-trained-models) for a list of the the external resources each entity type (annotated by the pre-trained models) is grounded to.

!!! note
    If you are using the web-service, simply supply `"ground": true` in your `JSON` payload to ground entities.

### Working with annotations

The `Saber.annotate()` method returns a simple `dict` object

```python
ann = saber.annotate("The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.")
```

which contains the keys `title`, `text` and `ents`

- `title`: contains the title of the article, if provided
- `text`: contains the text (which is minimally processed) the model was deployed on
- `ents`: contains a list of entities present in the `text` that were annotated by the model

For example, to see all entities annotated by the model, call

```python
ann['ents']
```

#### Converting annotations to JSON

The `Saber.annotate()` method returns a `dict` object, but can be converted to a `JSON` formatted string for ease-of-use in downstream applications

```python
import json

# convert to json object
json_ann = json.dumps(ann)

# convert back to python dictionary
ann = json.loads(json_ann)
```
