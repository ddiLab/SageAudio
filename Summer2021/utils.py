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

# ------------------------------------------------------------------------------
# Spotify.ipynb
def format_artist(artist):
    return (artist[2:-2].replace("\'", "")).title()

def select_artist(results1, artists1, results2, artists2):
    qflag1 = True
    qflag2 = True
    
    try:
        if not artists1.empty:
            artistnames1 = np.empty_like(artists1, dtype=tuple)
            cnt = 0
            for i in artists1:
                artistnames1[cnt] = (format_artist(i), cnt)
                cnt += 1
    except:
        artistnames1 = fixed("")
        qflag1 = False
     
    try:
        if not artists2.empty:
            artistnames2 = np.empty_like(artists2, dtype=tuple)
            cnt = 0
            for i in artists2:
                artistnames2[cnt] = (format_artist(i), cnt)
                cnt += 1
    except:
        artistnames2 = fixed("")
        qflag2 = False       
    
    if qflag1 and qflag2:
        interact(plot_double, results1=fixed(results1), results2=fixed(results2), Artist1=artistnames1, Artist2=artistnames2)
    elif qflag1:
        interact(plot_song, results=fixed(results1), Artist=artistnames1)
    elif qflag2:
        interact(plot_song, results=fixed(results2), Artist=artistnames2)
    else:
        print("No valid results.")
        
def plot_song(results, Artist):
    fig, ax=plt.subplots()

    stats = ('Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence')
    values = results.loc[:, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
    values = values.iloc[Artist]
    titlestring = "\"" + (results.iloc[Artist, 1]).title() + "\" by " + ((results.iloc[Artist, 5])[2:-2].replace("\'", "")).title()

    ax.barh(stats, values)
    ax.invert_yaxis()
    ax.set_xlabel('Values')
    ax.set_xlim(left=0.0, right=1.0)
    ax.set_title(titlestring)
    
def plot_double(results1, Artist1, results2, Artist2):
    stats = ('Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence')
    values1 = results1.loc[:, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
    values1 = values1.iloc[Artist1]
    title1 = "\"" + (results1.iloc[Artist1, 1]).title() + "\" by " + ((results1.iloc[Artist1, 5])[2:-2].replace("\'", "")).title()

    values2 = results2.loc[:, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
    values2 = values2.iloc[Artist2]
    title2 = "\"" + (results2.iloc[Artist2, 1]).title() + "\" by " + ((results2.iloc[Artist2, 5])[2:-2].replace("\'", "")).title()

    x = np.arange(len(stats))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width/2, values1, width, label='Song 1')
    rects2 = ax.barh(x + width/2, values2, width, label='Song 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.invert_yaxis()
    ax.set_title(title1 + " vs. " + title2)
    ax.set_yticks(x)
    ax.set_yticklabels(stats)
    ax.legend()
    ax.set_xlabel('Values')
    ax.set_xlim(left=0.0, right=1.0)
    
def process_query(tracks, query1, query2):
    queryresults = []
    for query in (query1, query2):
        # Check if empty
        if query == "":
            queryresults.append("")
            queryresults.append("")
        else:
            # Lower input
            name = query.lower()

            # Lower track names
            lowernames = tracks["name"]
            lowernames = lowernames.str.lower()
            tracks["name"] = lowernames

            # Lower artist names
            lowerartists = tracks["artists"]
            lowerartists = lowerartists.str.lower()
            tracks["artists"] = lowerartists

            # Search query
            songresults = tracks[tracks["name"] == name]
            artistlist = songresults["artists"]

            # Mark empty cases for exceptions
            if songresults.empty:
                songresults = ""
                artistlist = ""
            
            # Save results
            queryresults.append(songresults)
            queryresults.append(artistlist)
    
    # Test for empty results
    select_artist(queryresults[0], queryresults[1], queryresults[2], queryresults[3])
    
def spotify_display(tracks):
    input1 = widgets.Text(
        description='Song 1:',
        disabled=False
    )

    input2 = widgets.Text(
        description='Song 2:',
        placeholder="Optional",
        disabled=False
    ) 
    
    interact_manual(process_query, tracks=fixed(tracks), query1=input1, query2=input2, tooltip="Search")
    

#---------------------------------------------------
# Audio_Visualizer.ipynb
def select_options(song_waves, Song, Plot, Length):
    wave = song_waves[Song][0]
    sr = song_waves[Song][1]
    
    if Length == "Full song":
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
            select_options(song_waves, Song, Plot, "Full song")
            
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
    plt.show()

def plot_spectrogram(wave):
    D = librosa.stft(wave)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(18,6))
    lbdis.specshow(S_db, x_axis='s')
    plt.colorbar()
    
def option_interact(song_waves, song1, song2, plot1, plot2, length1, length2):
    plot1 = select_options(song_waves, song1, plot1, length1)
    plot2 = select_options(song_waves, song2, plot2, length2)

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
    length_box = ["Full song", "30 seconds", "5 seconds", "50 milliseconds"]

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

    out = widgets.interactive_output(option_interact, {'song_waves': fixed(song_waves), 'song1': song1, 'song2': song2, 'plot1': plot1, 'plot2': plot2, 'length1': length1, 'length2': length2})
    display(out)


# ---------------------------------------------------------------
# Data_Explorer.ipynb
def load_tracks():
    tracks = fma.load('data/fma_data/tracks.csv')
    tracks = tracks[tracks['set', 'subset'] <= 'small']
    artists = tracks['artist']
    tracks = tracks['track']
    
    # Search directory and make libraries
    subfolders = os.listdir('fma_small')
    track_files = []

    # Load mp3 filenames
    cnt = 0
    testvar = "empty"
    for i in subfolders:
        try:
            for j in os.listdir(f'fma_small/{i}'):
                testvar = f"fma_small/{i}/{j}"
                track_files.append(testvar)
        except:
            print(f"Skipped {i}")
            
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

    printstr = f"Track name: {track_name}\nGenre: {track_genre}\nArtist: {rand_artists}\nFilename: {track_file}"

    print(printstr)
    display(track_audio)
    
def explorer_display(tracks, track_files, artists):
    interact_manual(fma_random, tracks=fixed(tracks), track_files=fixed(track_files), artists=fixed(artists))
    
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
    cnt = np.arange(8000)
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

def genre_check(data):
    genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    count = len(data)
    print(f"Total: {count}")
    
    for i in genres:
        count = len(data.loc[data["genre_top"] == i, :])
        print(f"{i}: {count}")
    
