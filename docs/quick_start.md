# Quick Start

If your goal is simply to use Saber to annotate biomedical text, then you can either use the [web-service](#web-service) or a [pre-trained model](#pre-trained-models).

## Web-service

To use Saber as a **local** web-service, run:

```
(saber) $ python -m saber.app
```

or to build & run Saber with __Docker__:

```
# Build docker
(saber) $ docker build -t saber .

# Run docker (use `-it` instead of `-dt` to try it interactively)
(saber) $ docker run --rm -p 5000:5000 --name saber1 -dt saber
```

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

For example, running the web-service locally and using `cURL`:

```bash
(saber) $ curl -X POST 'http://localhost:5000/annotate/text' \
--data '{"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}'
```

Documentation for the Saber web-service API can be found [here](https://baderlab.github.io/saber-api-docs/). We hope to provide a live version of the web-service soon!

## Pre-trained models

First, import `SequenceProcessor`. This class coordinates training, annotation, saving and loading of models and datasets. In short, this is the interface to Saber.

```python
from saber.sequence_processor import SequenceProcessor
```

To load a pre-trained model, we first create a `SequenceProcessor` object

```python
sp = SequenceProcessor()
```

and then load the model of our choice

```python
sp.load('PRGE')
```

You can see all the pre-trained models in the [web-service API docs](https://baderlab.github.io/saber-api-docs/) or, alternatively, by running the following line of code

```python
from saber.constants import ENTITIES; print(list(ENTITIES.keys()))
```

To annotate text with the model, just call the `annotate()` method

```python
sp.annotate('A Sos-1-E3b1 complex directs Rac activation by entering into a tricomplex with Eps8.')
```

### Working with annotations

The `annotate()` method returns a simple `dict` object

```python
ann = sp.annotate('A Sos-1-E3b1 complex directs Rac activation by entering into a tricomplex with Eps8.')
```

which contains the keys `title`, `text` and `ents`:

- `title`: contains the title of the article, if provided
- `text`: contains the text (which is minimally processed) the model was deployed on
- `ents`: contains a list of entities present in the `text` that were annotated by the model

For example, to see all entities annotated by the model, call

```python
ann['ents']
```

#### Converting annotations to JSON

The `annotate()` method returns a `dict` object, but can be converted to a `JSON` formatted string for ease-of-use in downstream applications

```python
import json

# convert to json object
json_ann = json.dumps(ann)

# convert back to python dictionary
ann = json.loads(json_ann)
```
