# -------------------------------------------
# spotify.py
# Helper functions for Spotify.ipynb
# Emily Brown 8/13/2021
# -------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import IPython.display as ipy
import os
import pathlib

# --------------------------------------------
# Public functions

# get_tracks()
# Wrapper for pd.read_csv().
#
# Args: path - path of the .csv to read.
#
# Returns: pandas dataframe.

def get_tracks(path):
    return pd.read_csv(path)


# search()
# Creates user interface for running the song search.
#
# Args: tracks - dataframe containing song data to search.

def search(tracks):
    input1 = widgets.Text(
        description='Song 1:',
        disabled=False
    )

    input2 = widgets.Text(
        description='Song 2:',
        placeholder="Optional",
        disabled=False
    ) 
    
    out = interactive(process_query, {'manual': True}, tracks=fixed(tracks), query1=input1, query2=input2)
    out.children[2].description="Search"
    display(out)

    
    
# ------------------------------------------
# Private functions   
    
# ---
# process_query()
# Takes user search strings and querys the track data
# for it. Hands results of to select_artist() to continue processing
# results.
#
# Args: tracks - full dataset (pandas dataframe)
#       query1 - string of first song to search
#       query2 - string of second song to search

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

    
# ---
# select_artist()
# Prepares artists and tracks found in process_query() for plotting.
# Checks whether a given field was empty / returned no results, and
# selects the appropriate plotting configuration.
#
# Args: results1 - List of tracks found according to the first search.
#       artists1 - List of artists which had a song matching the first search. 
#       results2 - List of tracks found according to the second search.
#       artists2 - List of artists which had a song matching the second search.

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

        
# ---
# plot_song()
# Plots spotify feature metrics for a given song.
#
# Args: results - List of tracks found according to the search.
#       Artist - The artist of the specific song to plot.

def plot_song(results, Artist):
    fig, ax=plt.subplots()

    stats = ('Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence')
    values = results.loc[:, ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']]
    values = values.iloc[Artist]
    titlestring = "\"" + (results.iloc[Artist, 1]).title() + "\" by " + ((results.iloc[Artist, 5])[2:-2].replace("\'", "")).title()

    rects = ax.barh(stats, values)
    ax.invert_yaxis()
    ax.set_xlabel('Values')
    ax.set_xlim(left=0.0, right=1.0)
    ax.bar_label(rects, fmt='%.2f', padding=3)
    ax.set_title(titlestring)
    
    
# ---
# plot_double()
# Plots spotify features for two given songs on a single plot.
#
# Args: results1 - List of tracks found according to the first search.
#       Artist1 - The artist of the first specific song to plot.
#       results2 - List of tracks found according to the second search.
#       Artist2 - The artist of the second specific song to plot.

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
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    

# ---
# format_artist()
# Given an artist name, formats it for output by removing
# single quotes and enforcing capitalization.
#
# Args: artist - str of artist name to format
# 
# Returns: str of formatted artist name

def format_artist(artist):
    return (artist[2:-2].replace("\'", "")).title() 