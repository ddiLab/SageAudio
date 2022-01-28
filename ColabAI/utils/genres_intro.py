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
import fma
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
# Genres.ipynb
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
    feature_count = int(chroma) + int(rmse) + int(spectral_centroid) + int(spectral_bandwith) + int(spectral_rolloff) + int(zero_crossing_rate) + int(mfcc)
    mask = np.array([True, chroma, rmse, spectral_centroid, spectral_bandwith, spectral_rolloff, zero_crossing_rate], dtype=bool)

    if mfcc:
        mfcc_mask = np.ones(20, dtype=bool)
    else:
        mfcc_mask = np.zeros(20, dtype=bool)
        
    mask = np.concatenate((mask, mfcc_mask))
    mask = np.append(mask, True)
    new_features = features.loc[new_tracks.index, mask]
    print(f"{feature_count} features extracted for {len(new_tracks)} songs.")
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

def select_split():
    x_widget = widgets.IntSlider(min=0, max=100, step=10, value=10)
    y_widget = widgets.IntSlider(min=0, max=100, step=10, value=10)
    z_widget = widgets.IntSlider(min=0, max=100, step=10, value=10)

    def update_x_range(*args):
        x_widget.max = 100 - y_widget.value - z_widget.value
    x_widget.observe(update_x_range, 'value')
    y_widget.observe(update_x_range, 'value')
    z_widget.observe(update_x_range, 'value')

    def update_y_range(*args):
        y_widget.max = 100 - x_widget.value - z_widget.value
    x_widget.observe(update_y_range, 'value')
    y_widget.observe(update_y_range, 'value')
    z_widget.observe(update_y_range, 'value')

    def update_z_range(*args):
        z_widget.max = 100 - x_widget.value - y_widget.value
    x_widget.observe(update_z_range, 'value')
    y_widget.observe(update_z_range, 'value')
    y_widget.observe(update_z_range, 'value')

    def printer(Training, Validation, Test):
        total = Training + Validation + Test
        print(f"Total: {total}")
    interact(printer,Training=x_widget, Validation=y_widget, Test=z_widget);
    
    return x_widget, y_widget, z_widget
    
def select_model():
    layer_widget = widgets.IntSlider(min=2, max=8, step=1, value=4, description="Layers")
    epoch_widget = widgets.IntSlider(min=1, max=500, step=1, value=100, description="Epochs")
    
    def update_layer_range(*args):
        if epoch_widget.value > 350:
            layer_widget.max = 5
        elif epoch_widget.value > 100:
            layer_widget.max = 6
        elif epoch_widget.value > 25:
            layer_widget.max = 7
        else:
            layer_widget.max = 8
    epoch_widget.observe(update_layer_range, 'value')

    def update_epoch_range(*args):
        if layer_widget.value == 8:
            epoch_widget.max = 25
        elif layer_widget.value == 7:
            epoch_widget.max = 100
        elif layer_widget.value == 6:
            epoch_widget.max = 350
        else:
            epoch_widget.max = 500
    layer_widget.observe(update_epoch_range, 'value')
    display(layer_widget, epoch_widget)
   
    return layer_widget, epoch_widget
    
def display_dataset(trim_tracks):
    # Run data builder
    w = interactive(build_dataset, {'manual': True}, tracks=fixed(trim_tracks),
                electronic=widgets.IntSlider(min=0, max=999, value=999),
                experimental=widgets.IntSlider(min=0, max=999, value=999),
                folk=widgets.IntSlider(min=0, max=1000, value=1000),
                hiphop=widgets.IntSlider(min=0, max=1000, value=1000),
                instrumental=widgets.IntSlider(min=0, max=1000, value=1000),
                international=widgets.IntSlider(min=0, max=1000, value=1000),
                pop=widgets.IntSlider(min=0, max=1000, value=1000),
                rock=widgets.IntSlider(min=0, max=999, value=999));
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

def select_dataset():
    dataset_widget = widgets.Dropdown(
        options=[(1, 'Electronic'), (2, 'Experimental'), 
                 (3, 'Folk'), (4, 'Hip-Hop'),
                 (5, 'Instrumental'), (6, 'International'),
                 (7, 'Pop'), (8, 'Rock'), (9, 'Full')],
        #value=1,
        description='Dataset: ')
    
    display(dataset_widget)
    return dataset_widget

def param_check(training, validation, test):
    check = training + validation + test
    if check != 100:
        raise Exception(f"Error: Training, validation, and test splits must add up to 100. Current total: {check}")

def preprocessing(features, genre):
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
    
    # Build list of indeces w/ target genre
    if genre != "Full":
        cnt = 0
        mask = np.ones(len(y_train), dtype=bool)

        for i in y_train.flat:
            if i != (encoder.transform([genre]))[0]:
                mask[cnt] = False
            cnt += 1
    
        X_train = X_train[mask]
        y_train = y_train[mask]
    
    train_data = [X_train, y_train]
    test_data = [X_test, y_test]
    return train_data, test_data, encoder


def build_and_train_model(train_data):
    model = build_model(train_data)
    train_model(model, train_data)
    return model
    

def build_model(train_data, size):
    GENRE_CNT = 8
    
    # calculate layer sizes
    sizes = np.full(size - 1, 2)
    cnt = 0
    for i in sizes:
        sizes[cnt] = i**(cnt + 6)
        cnt += 1

    # build model
    model = Sequential()
    cnt -= 1
    model.add(layers.Dense(sizes[cnt], activation='relu', input_shape=(train_data[0].shape[1],)))
    cnt -= 1
    while cnt >= 0:
        model.add(layers.Dense(sizes[cnt], activation='relu'))
        #print(sizes[cnt])
        cnt -= 1
        
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
    
def test_model(model, test_data, encoder):
    # Run predictions
    print('Overall Results:')
    test_scores = model.evaluate(test_data[0], test_data[1], verbose=2)
    predictions = model.predict(test_data[0][:])

    # Save predictions
    results = []
    nums = []
    for i in predictions:
        highest = 0
        cnt = 0
        for j in i:
            if j > i[highest]:
                highest = cnt
            cnt += 1
        results.append(highest)
        nums.append(i[highest])

    # Mark successes and failures
    success = []
    cnt = 0
    for i in results:
        if i == test_data[1][cnt]:
            success.append("SUCCESS")
        else:
            success.append("FAILURE")
        cnt += 1
    
    # Print individual results
    results = encoder.inverse_transform(results)
    print('\nPredictions:')
    cnt = 0
    while cnt < 10:
        print('Song {}'.format(cnt + 1))
        print('Prediction: {}'.format(results[cnt]))
        print('Confidence: {:.2%}'.format(nums[cnt]))
        print('Result: {}\n'.format(success[cnt]))
        cnt += 1