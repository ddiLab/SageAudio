# Summer2021
06/2021 - Current project  
Educational notebooks written for summer 2021 camp on basic ML/AI concepts and their applications in scientific research. Notebooks are written to be as clean and accessible as possible for high schools students who may not have any programming experience, heavily utilizing the [ipywidgets](https://pypi.org/project/ipywidgets/) library. Various notebooks written during spring 2021 for AudioBasics and MusicGenre have been adapted here.

## Goals
### Finished goals:
- 3 notebooks prototyped:
-   Spotify.ipynb
-   Audio_Visualizer.ipynb
-   Data_Explorer.ipynb

### Current / future goals:
- Clean, document, and finalize prototype notebooks
- Prototype remaining week 1/2 notebooks
- Begin work on week 3/4 notebooks

## Files and Implementation notes
### Dependencies
The current environment (`environment.yml`) is a work in progress that hodge-podges environments and libraries I've previously used for audio projects and will be updated and cleaned up as needed. Similar to the AudioBasics and MusicGenre environments, this environment includes the Librosa library for audio feature extraction, meaning that these notebooks requre the [ffmpeg](https://www.ffmpeg.org/) tool.

### Data
#### FMA

#### Spotify

### Notebooks  
- **`Spotify.ipynb`** (cleaning stage) - Searches a database of Spotify songs and plots various high level features tracked by Spotify for each song.
- **`Audio_Visualizer.ipynb`** (cleaning stage) - Plots visualizations of input audio files such as waveform plot, spectrum diagram, and spectrogram.
- **`Data_Explorer.ipynb`** (cleaning stage) - Pulls a random song from the FMA dataset, plays it, and lists basic metadata info of it.
- **`Datasets.ipynb`** (prototyping stage) - Builds subsets of FMA dataset for students to train models on.

### Files
- **`fma.py`** - Helper functions and classes from `utils.py` in [original FMA code](https://github.com/mdeff/fma) used in navigating the FMA dataset.
- **`utils.py`** - Library where actual functionality of the notebooks is being abstracted away to.
