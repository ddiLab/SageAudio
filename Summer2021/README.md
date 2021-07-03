# Summer2021
06/2021 - Current project  
Educational notebooks written for summer 2021 camp on basic ML/AI concepts and their applications in scientific research. Notebooks are written to be as clean and accessible as possible for high schools students who may not have any programming experience, heavily utilizing the [ipywidgets](https://pypi.org/project/ipywidgets/) library. Various notebooks written during spring 2021 for AudioBasics and MusicGenre have been adapted here.

## Goals
### Finished goals:
- 3 notebooks finished:
-   Spotify.ipynb
-   Audio_Visualizer.ipynb
-   Data_Explorer.ipynb
- 1 notebook prototyped:
-   Datasets.ipynb

### Current / future goals:
- Finish week 2 notebook(s)
- Begin work on week 3/4 notebooks

## Implementation notes
### Dependencies
Two options are provided in `environments`, one utilizing conda and one utilizing only pip. These environment includes the Librosa library for audio feature extraction, meaning that these notebooks requre the [ffmpeg](https://www.ffmpeg.org/) tool be installed.

### Data
#### FMA
The [Free Music Archive](https://github.com/mdeff/fma) is a dataset built by Michaël Defferrard, et al. meant to be an easily accessible resource for the study of music information retrieval. Consturcted entirely from Creative Commons-liscensed music, the full dataset consists of 106,574 tracks represented 161 genres. Here, we utilize a 7.2GB subset provided by Defferrard, et al. that consists of only 8000 tracks, each trimmed to 30 second clips, representing 8 genres (available [here](https://os.unil.cloud.switch.ch/fma/fma_small.zip)). We also utilize metadata information on the data from `tracks.csv` (available [here](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)), as well as features extracted from the FMA audio data (available [here](https://github.com/ddiLab/SageAudio/blob/main/MusicGenre/fma_small.csv)).

A known issue with this data is that 3 of the tracks in the small subset are corrupted and unusable, leaving us with 7997 tracks instead of 8000. The current working solution is to ignore these three tracks and work with the less pretty and slightly unbalanced total of 7997 tracks. WE may in the future look into substituting the corrupted tracks with suitable replacements from the larger subsets of the dataset.

#### Spotify
For `Spotify.ipynb`, we utilize [a Kaggle dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=tracks.csv) which contains metadata pulled from the Spotify API for approximately ~600k songs. We are particularly interested in high-level feature data Spotify tracks, including features such as danceability, acousticness, valence, etc. for introducing students to ideas of what song features might look like and how they might relate to a song's genre. We only utilize `tracks.csv` from this dataset.

#### Directory structure
Data for these notebooks should be kept in a directory titled `data` that is structured as such:
```
...fma
......(fma_small subdirectories)
...songs
......(audio files)
...fma_features.csv
...fma_tracks.csv
...spotify.csv
```
The `fma` directory should contain the contents of [`fma_small.zip`.](https://os.unil.cloud.switch.ch/fma/fma_small.zip) The `songs` directory should contain audio files to be used in `Audio_Visualizer.ipynb`. `fma_features.csv` is a [file containing the extracted audio features from the fma_small set.](https://github.com/ddiLab/SageAudio/blob/main/MusicGenre/fma_small.csv). `fma_tracks.csv` is a renamed version of `tracks.csv` from [`fma_metadata.zip`.](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip). `spotify.csv` is a renamed version of `tracks.csv` from the [Spotify Kaggle dataset.](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=tracks.csv)  

## Contents
- **`Spotify.ipynb`** (finished) - Searches a database of Spotify songs and plots various high level features tracked by Spotify for each song.
- **`Audio_Visualizer.ipynb`** (finished) - Plots visualizations of input audio files such as waveform plot, spectrum diagram, and spectrogram.
- **`Data_Explorer.ipynb`** (finished) - Pulls a random song from the FMA dataset, plays it, and lists basic metadata info of it.
- **`Datasets.ipynb`** (prototype) - Builds subsets of FMA dataset for students to train models on.

### `utils`
- Each notebook has a corresponding `.py` file in `utils/` that contains the bulk of the actual code for the given notebook.
- **`fma.py`** - Helper functions and classes from `utils.py` in [original FMA code](https://github.com/mdeff/fma) used in navigating the FMA dataset.

### `environments`
- **`environment.yml`** - Environment code was originally written for, uses a combination of conda and pip installs. If anything breaks really hard (which it shouldn't), try using this environment over the other one.
- **`requirements.txt`** - Environment based off of `environments.yml` that only uses pip, no conda neededd.

