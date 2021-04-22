# MusicGenre
03/2021 - Current project  
Explorations in machine learning and music information retrieval focused on the task of genre classification. This work is based largely on two sources, utilizing the [FMA dataset](https://github.com/mdeff/fma) built by Defferrard, et al., as well as methods and code from a [KDnuggets article on the subject](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html) by Nagesh Singh Chauhan. The goal with this project is to use the work put forth by these two sources as a base for building simple notebooks on machine learning-powered music genre classification that can be used to teach (especially to unfamiliar high school students) the basic concepts of machine learning.  

## Goals
### Finished goals:
- Successfully build and run genre classification models using FMA and GTZAN datasets
- Extract features from small subset of FMA dataset using method from [Chauhan's article](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html) (`Features.ipynb`)
- Extract features from user-submitted test audio and compare it to a model trained on fma_small features (`Genres.ipynb`)

### Future goals:
- Extract features from larger FMA subset to hopefully improve model in `Genres.ipynb`
- Write version of `Genres.ipynb` that uses raw audio to train model as opposed to extracted audio features
- Write version of `Genres.ipynb` that further abstracts away the implementation to make more understandable for unfamiliar high school students, etc.

## Files and Implementation notes
### Dependencies
The environment used here (`environment.yml`) is based closely off the environment used in [Chauhan's article](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html). Notably, this environment includes the Librosa library for audio feature extraction, meaning that these notebooks require the user to have [ffmpeg](https://www.ffmpeg.org/) installed.

### Notebooks  
- **`Genres.ipynb`** (04/2021) - Based off of [Chauhan's article](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html). Given the extracted audio features from the FMA small subset (`fma_small.csv`), builds a genre classification model, extracts features from user-submitted test audio, and makes genre predictions using the test audio features. Currently, the model doesn't perform very well (~45% accuracy after 100 epochs), possibly due to the small dataset (8000 30s tracks).
- **`Features.ipynb-`** (04/2021) - Based off of [Chauhan's article](https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html). Extracts features from the FMA small subset and makes `fma_small.csv`.

### Other files
- **`fma_small.csv`** - Audio features extracted from the small subset of the FMA dataset, created using `Features.ipynb`. The extraction process can take a while to run, hence why I've included the end result here.
- **`utils.py`** - Helper functions and classes from [original FMA code](https://github.com/mdeff/fma), used in `Features.ipynb`.
