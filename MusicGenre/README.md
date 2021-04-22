# FMA
03/2021 - Current project  
Explorations in machine learning and music information retrieval based on [paper](https://arxiv.org/pdf/1612.01840.pdf) and [code](https://github.com/mdeff/fma) written by Defferrard, et al. The goal with this project is to use the work by Defferrard, et al. as a base for building simple notebooks on machine learning-powered music genre detection that can be used to teach (especially to unfamiliar high school students) the basic concepts of machine learning.  

## Dependencies  
Currently, I'm using a scaled-down version of the environment used in the original code. An `environment.yml` file is provided. Additionally, the notebooks I've written so far currently only use the FMA metadata and none of the raw audio data. [`fma_metadata.zip`](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) can be downloaded here.

**Current goal:** Reconfigure the code from Defferrard, et al. used to extract audio features from `.mp3` files into formatted `.csv` files into a way to easily upload a sample `.mp3` and have the model classify it.
**Other goals:** Look into other model types to use, look into other ways of training on audio data (aside from `.csv`-extracted features), clean up and document everything.

## Notebooks  
- **`Note Generator.ipynb`** (03/2021) - My first success in getting the model to build and work on my own machine using a scaled-down environment and only .csv metadata, no raw audio data. Adapted from `usage.ipynb` from [original code](https://github.com/mdeff/fma). Needs more documenting.

## Other files
- **`utils.py`** - Helper functions and classes written for [original code](https://github.com/mdeff/fma), entirely unaltered.
