# Audio Basics  
01/2021 - 02/2021, 06/2021
Simple notebooks and scripts that demonstrate various basic tasks in audio processing and visualization. These were built both as excercises in learning about audio processing in Python and Jupyter as well as to be teaching tools for the same purpose. Two environments are provided, one for the two Jupyter Notebooks and one for the Raspberry Pi python script.

## Notebooks  
Two notebooks are provided:
- **`Note_Generator.ipynb`** (01/2021) - A simple notebook which uses interactive widgets to build and visualize a sound wave. It generates up to four soundwaves and adds them together. For each soundwave, the waveform, pitch, and octave can be adjusted, as can the length of the full soundwave. Additionally, the full soundwave can be visualized both in the time domain as a waveform and in the frequency domain as a spectrum. This notebook utilizes [thinkdsp](https://greenteapress.com/wp/think-dsp/), a library designed as a teaching tool for audio processing that streamlines numpy, scipy, and matplotlib functionalities for ease of understanding. `thinkdsp.py` is provided in this folder.
- **`Webcam_Recorder.ipynb`** (02/2021) - Records a short audio clip using a webcam microphone, writing it to a wave file and visualizing a waveform and a spectogram. Both the initial .webm output file and the converted .wav file are saved as `record` to subdirectory `library.` In addition to the environment provided here, this notebook and its dependencies also require [ffmpeg](https://www.ffmpeg.org/).
- **`Audio_Visualizer.ipynb`** (06/2021) - Loads audio files from a local directory `songs` and draws various visualizations of them, including waveform plots, spectrum diagrams, and spectrograms. Two visualizations can be drawn at a time for the sake of comparison, and each visualization can be "zoomed in" to smaller timeframes of the audio file (30 seconds, 5 seconds, 50 milliseconds). Built for Summer 2021 education intiative. Needs further cleaning and documentation.   

## Scripts
One script is provided:
- **`usbaudio.py`** (02/2021) - Script built for Raspberry Pi that records audio from a usb microphone and optionally plots various visualizations. Defaults to recording 5 seconds of audio and saving it to `out.wav` and no visualizations. Can visualize two versions of time domain (full waveform or zoomed in waveform) and frequency domain. See docbox in file for a detailed description of the available command line options. Notably does **not** require ffmpeg, utilizing a small number of libraries.
