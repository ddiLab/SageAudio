import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import IPython.display as ipy
import os
import pathlib
import librosa
from utils import fma
import random
from librosa import display as lbdis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

#---------------------------------------------------
# Audio_Visualizer.ipynb
def select_options(song_waves, Song, Plot, Length):
    wave = song_waves[Song][0]
    sr = song_waves[Song][1]
    
    if Length == "Full":
        if Plot == "Wave":
            plot_wave(wave)
        elif Plot == "Spectrum":
            plot_spectrum(wave, sr)
        elif Plot == "Spectrogram":
            plot_spectrogram(wave)           
    else:
        if Length == "30 seconds":
            gen_length = 30 * sr
        if Length == "5 seconds":
            gen_length = 5 * sr
        if Length == "50 milliseconds":
            gen_length = int(0.05 * sr)
            
        start_length = 30 * sr
        if gen_length < len(wave):
            wave_crop = wave
            wave_crop = wave_crop[(int(len(wave_crop)/2)-int(gen_length/2)):(int(len(wave_crop)/2)+int(gen_length/2))]
            if Plot == "Wave":
                plot_wave(wave_crop)
            elif Plot == "Spectrum":
                plot_spectrum(wave_crop, sr)
            elif Plot == "Spectrogram":
                plot_spectrogram(wave_crop) 
        else:
            select_options(song_waves, Song, Plot, "Full")
            
def plot_wave(wave):
    plt.figure(figsize=(18,6))
    lbdis.waveplot(wave, alpha=0.5)

def plot_spectrum(wave, sr):
    # derive spectrum using FT
    ft = sp.fft.fft(wave)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sr, len(magnitude))

    # plot spectrum
    plt.figure(figsize=(18,6))
    plt.plot(frequency[:(int(len(frequency)/2))], magnitude[:(int(len(magnitude)/2))]) # magnitude spectrum
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

def plot_spectrogram(wave):
    D = librosa.stft(wave)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(18,6))
    lbdis.specshow(S_db, x_axis='s')
    #plt.colorbar()
    
def option_interact(song_waves, song_names, song1, song2, plot1, plot2, length1, length2):
    select_options(song_waves, song1, plot1, length1)
    plt.title(f"{plot1} plot for \"{song_names[song1]}\" ({length1})")
    
    select_options(song_waves, song2, plot2, length2)
    plt.title(f"{plot2} plot for {song_names[song2]} ({length2})")
    
def find_songs(path):    
    # Establish input directory
    input_dir = pathlib.Path(path)
    if not input_dir.exists():
        os.mkdir(path)
    
    # Search directory and make libraries
    input_content = os.listdir(input_dir)
    song_waves = np.empty_like(input_content, dtype=np.ndarray)
    song_names = np.empty_like(input_content)
    song_box = np.empty_like(input_content, dtype=tuple)

    cnt = 0
    for i in input_content:
        song_waves[cnt] = librosa.load(f'{path}/{i}')
        song_names[cnt] = i[:-4]
        song_box[cnt] = (i[:-4], cnt)
        cnt += 1
        
    return [song_waves, song_names, song_box]
    
def visualizer_display(song_waves, song_names, song_box):
    plot_box = ["Wave", "Spectrum", "Spectrogram"]
    length_box = ["Full", "30 seconds", "5 seconds", "50 milliseconds"]

    song1 = widgets.Dropdown(options=song_box, description="Songs: ")
    song2 = widgets.Dropdown(options=song_box)
    song_ui = widgets.HBox([song1, song2])
    display(song_ui)

    plot1 = widgets.Dropdown(options=plot_box, description="Plots: ")
    plot2 = widgets.Dropdown(options=plot_box)
    plot_ui = widgets.HBox([plot1, plot2])
    display(plot_ui)

    length1 = widgets.Dropdown(options=length_box, description="Length: ")
    length2 = widgets.Dropdown(options=length_box)
    length_ui = widgets.HBox([length1, length2])
    display(length_ui)

    out = widgets.interactive_output(option_interact, {'song_waves': fixed(song_waves), 'song_names': fixed(song_names), 'song1': song1, 'song2': song2, 'plot1': plot1, 'plot2': plot2, 'length1': length1, 'length2': length2})
    display(out)
