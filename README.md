# tennisML

This repository contains two python scripts which have various functions and tools used for tennis modelling: tennisML.py contains the functions used to wrangle and transform data, runscript.py gives an example of how these functions can be used with the XGBoost algorithm to generate predictions for the 2020 Australian Open.

After cloning and changing directory into tennis ML, type: python runscript.py, and hit enter. This will run the demo script producing average predicted win probabilities for the 2020 Australian Open.

```
python runscript.py
```

## Documentation
First auto-generate the *.rst files
```
sphinx-apidoc --force -o docs/_modules . Data
```
