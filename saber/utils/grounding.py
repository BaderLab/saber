import requests

## TODO: use UniProt REST api: search
# Example:
#  https://www.uniprot.org/uniprot/?query=name:mk2%20OR%20gene:mk2%20OR%20mnemonic:MK2&sort=score&columns=id,organism-id,genes(PREFERRED)&format=tab
#
# Search fields, result columns, etc, see:
#  https://www.uniprot.org/help/query-fields
#  https://www.uniprot.org/help/uniprotkb_column_names
#
def uniprot_search(text, fields=('name','gene','mnemonic'), columns=('id','organism-id','genes(PREFERRED)')):
    return {}

#TODO: describe
def hgnc_search(text):
    return {}
