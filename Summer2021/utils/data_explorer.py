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

# ---------------------------------------------------------------
# Data_Explorer.ipynb
def load_tracks(path, directory):
    tracks = fma.load(path)
    tracks = tracks[tracks['set', 'subset'] <= 'small']
    artists = tracks['artist']
    tracks = tracks['track']
    
    # Search directory and make libraries
    subfolders = os.listdir(directory)
    track_files = []

    # Load mp3 filenames
    cnt = 0
    testvar = "empty"
    for i in subfolders:
        try:
            for j in os.listdir(f'{directory}{i}'):
                track_files.append(f"{directory}{i}/{j}")
        except:
            pass
            
    return tracks, track_files, artists
    
def fma_random(tracks, track_files, artists):
    #ipy.clear_output()
    #display(button)
    
    rand_index = random.randint(0, 7999)
    rand_track = tracks.iloc[rand_index]
    rand_artists = artists.iloc[rand_index, 12]
    track_name = tracks.iloc[rand_index, 19]
    track_genre = tracks.iloc[rand_index, 7]
    track_file = track_files[rand_index]
    track_audio = ipy.Audio(track_files[rand_index])

    printstr = f"Track name: {track_name}\nArtist: {rand_artists}\nGenre: {track_genre}\nFilename: {track_file}"

    print(printstr)
    display(track_audio)
    
def interface(tracks, track_files, artists):
    out = interactive(fma_random, {'manual': True}, tracks=fixed(tracks), track_files=fixed(track_files), artists=fixed(artists))
    out.children[0].description="Search"
    display(out)