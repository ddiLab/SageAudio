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

# -----------------------------------------------------------------
# Datasets.ipynb
def build_dataset(tracks, electronic, experimental, folk, hiphop, instrumental, international, pop, rock):
    # Read in genre counts
    genre_counts = {"Electronic": electronic, 
                "Experimental": experimental,
                "Folk": folk,
                "Hip-Hop": hiphop,
                "Instrumental": instrumental,
                "International": international,
                "Pop": pop,
                "Rock": rock
               }
    
    # Build list of tracks to delete
    cnt = np.arange(7997)
    indeces = tracks.index
    delete_indeces = []
    for i in cnt:
        track = tracks.iloc[i, :]
        genre = track["genre_top"]
        if genre_counts[genre] > 0:
            genre_counts[genre] = genre_counts[genre] - 1
        else:
            delete_indeces.append(i)
            
    # Build trimmed copy
    indeces = indeces.delete(delete_indeces)
    new_tracks = tracks.loc[indeces, :]
    genre_check(new_tracks)
    return new_tracks

def extract_features(tracks, new_tracks, features, chroma, rmse, spectral_centroid, spectral_bandwith, spectral_rolloff, zero_crossing_rate, mfcc):
    mask = np.array([True, chroma, rmse, spectral_centroid, spectral_bandwith, spectral_rolloff, zero_crossing_rate], dtype=bool)

    if mfcc:
        mfcc_mask = np.ones(20, dtype=bool)
    else:
        mfcc_mask = np.zeros(20, dtype=bool)
        
    mask = np.concatenate((mask, mfcc_mask))
    mask = np.append(mask, True)
    new_features = features.loc[new_tracks.index, mask]
    display(new_features)
    return new_features

def genre_check(data):
    genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    count = len(data)
    print(f"Total: {count}")
    
    for i in genres:
        count = len(data.loc[data["genre_top"] == i, :])
        print(f"{i}: {count}")
        
def trim_data(tracks):
    mask = np.ones(8000, dtype=bool)
    d_mask = pd.Series(mask, index=tracks.index)
    d_mask[99134] = False
    d_mask[108925] = False
    d_mask[133297] = False
    return tracks[d_mask]

def prep_features(features, tracks):
    new_features = features.set_index(tracks.index)
    return new_features

def load_metadata(path):
    # Load track and artist data
    full = fma.load(path)
    small = full[full['set', 'subset'] <= 'small']
    artists = small['artist']
    tracks = small['track']
    trim_tracks = trim_data(tracks)
    return artists, tracks, trim_tracks

def load_features(path, trim_tracks):
    # Load feature data
    features = pd.read_csv(path)
    features = prep_features(features, trim_tracks)
    return features

def display_dataset(trim_tracks):
    # Run data builder
    w = interactive(build_dataset, {'manual': True}, tracks=fixed(trim_tracks),
                electronic=widgets.IntSlider(min=0, max=999),
                experimental=widgets.IntSlider(min=0, max=999),
                folk=widgets.IntSlider(min=0, max=1000),
                hiphop=widgets.IntSlider(min=0, max=1000),
                instrumental=widgets.IntSlider(min=0, max=1000),
                international=widgets.IntSlider(min=0, max=1000),
                pop=widgets.IntSlider(min=0, max=1000),
                rock=widgets.IntSlider(min=0, max=999));
    w.children[8].description="Build dataset"
    display(w)
    return w

def display_features(trim_tracks, dataset, features):
    # Run feature selector
    z = interactive(extract_features, {'manual': True},
                tracks=fixed(trim_tracks), new_tracks=fixed(dataset),
                features=fixed(features),
                chroma=False,
                rmse=False,
                spectral_centroid=False,
                spectral_bandwith=False,
                spectral_rolloff=False,
                zero_crossing_rate=False,
                mfcc=False);
    z.children[7].description="Extract features"
    display(z)
    return z

def preprocessing(features):
    # Preprocess data for model
    GENRE_LIST = 'electronic experimental folk hip-hop instrumental international pop rock'.split()
    GENRE_CNT = 8

    # Load features and trim filename column
    #data = pd.read_csv(FEATURES)
    data = features
    data = data.drop(['filename'],axis=1)

    # Encoding the labels
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    # Scaling the feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    # Dividing data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_data = [X_train, y_train]
    test_data = [X_test, y_test]
    return train_data, test_data

def build_model(train_data):
    GENRE_CNT = 8
    
    # Building the model
    model = Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(train_data[0].shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(GENRE_CNT, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def train_model(model, train_data):
    # Fit the model
    classifier = model.fit(train_data[0],
                    train_data[1],
                    epochs=100,
                    batch_size=128)
    
def test_model(model, test_data):
    # Test set
    print('Results:')
    test_scores = model.evaluate(test_data[0], test_data[1], verbose=2)
    predictions = model.predict(test_data[0][:])