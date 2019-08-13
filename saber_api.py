from saber import Saber

# Train is stateless, just needs a filepath to a config
Saber.train(config)

## Annotation
saber = Saber()
saber.load(model_or_list_of_models)

# A small function will be used to determine if the input is text, a dir, or a pmid
saber.annotate(text_dir_or_pmid, title, jupyter, coref, ground)