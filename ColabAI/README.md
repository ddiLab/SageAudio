# ColabAI
Spring 2022

Educational notebooks originally written for summer 2021 camp on basic ML/AI concepts and their applications in scientific research. Notebooks are written to be as clean and accessible as possible for high schools students who may not have any programming experience, heavily utilizing the [ipywidgets](https://pypi.org/project/ipywidgets/) library. Various notebooks written during spring 2021 for AudioBasics and MusicGenre have been adapted here.

This repository is adapted from the original [Summer2021](https://github.com/ddiLab/SageAudio/tree/main/Summer2021) repo, allowing educators to set up their own instances of the camp in a shared Google Drive so that students can interact with the notebooks using Google Colab.

## Usage
### Setting up the shared drive
The contents of this repo are intended to be downloaded into a shared drive on Google Drive, along with all data used by the notebooks. The directory structure of the shared drive should be set up like this:

```
Shared Drive root
├── data
|   ├── audio
|   |   └── (audio files as desired)
|   ├── fma_features.csv
|   ├── fma_tracks.csv
|   └── spotify.csv
├── fma
|   └── (fma_small subdirectories)
├── utils
|   ├── audio_visualizer.py
|   ├── data_explorer.py
|   ├── fma.py
|   ├── genres_full.py
|   ├── genres_intro.py
|   └── spotify.py
├── Audio_Visualizer.ipynb
├── Data_Explorer.ipynb
├── Genres_Full.ipynb
├── Genres_Intro.ipynb
└── Spotify.ipynb
```

Essentially, it should look identical to this repo, with additional folders `data` and `fma` holding data for the notebooks to access.

Once the shared drive has been set up properly, students can be given view access to it. They can then save copies of the notebooks (.ipynb files) which they can run on their own in Google Colab.

### Accessing shared drive data within notebooks
In the initial Colab run of this camp, the directory name of the shared drive we used was `AI-data[SharedDrive]`. If the shared drive you are using has a different name, then references to this name within the notebooks must be changed.

Each notebook begins with a cell that connects it to the user's Google Drive, followed by a cell that imports all code and data it needs from the shared drive. This second cell (third in `Spotify.ipynb`) is the one you will need to edit. Any occurence of `AI-data[SharedDrive]` in this cell must be replaced with the name of your shared drive. For example, if my shared drive were named `my_example_drive`, this is what I would have to change the second cell in `Audio_Visualizer.ipynb` to:

```
import sys
sys.path.append('drive/Shareddrives/my_example_drive/utils')
import audio_visualizer as av
songs = av.find_songs("drive/Shareddrives/my_example_drive/data/audio") # Path to audio file directory
```

## Data
### FMA
The [Free Music Archive](https://github.com/mdeff/fma) is a dataset built by Michaël Defferrard, et al. meant to be an easily accessible resource for the study of music information retrieval. Consturcted entirely from Creative Commons-liscensed music, the full dataset consists of 106,574 tracks represented 161 genres. Here, we utilize a 7.2GB subset provided by Defferrard, et al. that consists of only 8000 tracks, each trimmed to 30 second clips, representing 8 genres (available [here](https://os.unil.cloud.switch.ch/fma/fma_small.zip)). We also utilize metadata information on the data from `tracks.csv` (available [here](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)), as well as features extracted from the FMA audio data (available [here](https://github.com/ddiLab/SageAudio/blob/main/MusicGenre/fma_small.csv)).

A known issue with this data is that 3 of the tracks in the small subset are corrupted and unusable, leaving us with 7997 tracks instead of 8000. The current working solution is to ignore these three tracks and work with the less pretty and slightly unbalanced total of 7997 tracks. We may in the future look into substituting the corrupted tracks with suitable replacements from the larger subsets of the dataset.

### Spotify
For `Spotify.ipynb`, we utilize [a Kaggle dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=tracks.csv) which contains metadata pulled from the Spotify API for approximately ~600k songs. We are particularly interested in high-level feature data Spotify tracks, including features such as danceability, acousticness, valence, etc. for introducing students to ideas of what song features might look like and how they might relate to a song's genre. We only utilize `tracks.csv` from this dataset.

### Directory structure
Data for these notebooks should be kept in a two separate directories in the shared drive, as follows:
```
data
├── audio
|   └── (audio files as desired)
├── fma
|   └── (fma_small subdirectories)
├── fma_features.csv
├── fma_tracks.csv
└── spotify.csv

```
The `fma` directory should contain the contents of [`fma_small.zip`.](https://os.unil.cloud.switch.ch/fma/fma_small.zip) The `audio` directory should contain audio files to be used in `Audio_Visualizer.ipynb`. `fma_features.csv` is a [file containing the extracted audio features from the fma_small set.](https://github.com/ddiLab/SageAudio/blob/main/MusicGenre/fma_small.csv). `fma_tracks.csv` is a renamed version of `tracks.csv` from [`fma_metadata.zip`.](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip). `spotify.csv` is a renamed version of `tracks.csv` from the [Spotify Kaggle dataset.](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks?select=tracks.csv)  

## Contents
- **`Audio_Visualizer.ipynb`** - Plots visualizations of input audio files such as waveform plot, spectrum diagram, and spectrogram.
- **`Data_Explorer.ipynb`** - Pulls a random song from the FMA dataset, plays it, and lists basic metadata info of it.
- **`Genres_Full.ipynb`** - Interactively builds a dataset from FMA data, then builds, trains, and runs a model using it.
- **`Genres_Intro.ipynb`** - Abridged version of `Genres_Full.ipynb` in which students are secretly given intentionally skewed datasets.
- **`Spotify.ipynb`** - Searches a database of Spotify songs and plots various high level features tracked by Spotify for each song.

### `utils`
- Each notebook has a corresponding `.py` file in `utils/` that contains the bulk of the actual code for the given notebook.
- **`fma.py`** - Helper functions and classes from `utils.py` in [original FMA code](https://github.com/mdeff/fma) used in navigating the FMA dataset.

