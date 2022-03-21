# EdgeAudio
12/2021 - Current project  
Exploring audio domain problems in an edge computing environment. This projetc so far has been concerned with Nvidia Jetson Nano platform, and as such is based on work from [SageEdu](https://github.com/ddiLab/SageEdu).

## Goals
### Finished goals:
- Finished first prototype for an audio-detecting sound recorder notebook

### Future goals:
- Expand on `EventRecorder.ipynb` prototype
- Connect utility developed for `EventRecorder.ipynb` to ML bird classification
- Explore other audio domain problems in edge computing

## Files and Implementation notes
### Environment
The environment used for this code was set up according to the guide from [SageEdu](https://github.com/ddiLab/SageEdu/tree/main/setup/general) on building a suitable environment for Jupyter Notebooks on the Jetson Nano. This code additionally utilizes the [PyWaggle](https://github.com/waggle-sensor/pywaggle) library. 

### Notebooks  
- **`EventRecorder.ipynb`** (12/2021) - First prototype for an edge computing-oriented notebook which "smartly" records audio by distinguishing between useful sound information and background noise. Based on work from SageEdu utilizing PyWaggle, it continuously records audio while only saving what it determines to be audio "events" by doing simple energy-level analysis on what it has recorded.
