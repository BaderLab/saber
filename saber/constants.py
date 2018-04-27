"""Collection of constants used by Saber."""

# colours for use with displacy
COLOURS = {'PRGE': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)',
           'CHED': 'linear-gradient(90deg, #00E676, #B9F6CA)',
           'LIVB': 'linear-gradient(90deg, #F06292, #F8BBD0)',
           'CL': 'linear-gradient(90deg, #64B5F6, #BBDEFB)',
          }
# entity options for use with displacy
OPTIONS = {'ents': ['PRGE', 'CHED', 'LIVB', 'CL', 'SO', 'GO'],
           'colors': COLOURS
          }
