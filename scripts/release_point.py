# ----- IMPORTS/SETUP -----

# Import sklearn/tensorflow modules.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# Import other modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import statsmodels.formula.api as smf
import pickle
import math
import os
from time import sleep
from sportsipy.mlb.teams import Teams
from IPython.display import clear_output
from datetime import date

# Test dataset.
dataset = pd.read_csv('./data/data.csv')

# Color key.
pitch_color_key = {
    'Changeup': 'blue',
    'Curveball': 'green',
    'Cutter': 'brown',
    'Four-Seam': 'red',
    'Other': 'gray',
    'Sinker': 'orange',
    'Slider': 'pink',
    'Splitter': 'skyblue'
}

# ----- GET DATASET -----

def get_dataset():
    dataset_path = input('Dataset path > ')

    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print('File not found. Please try again.')
        sleep(2)
        get_dataset()


get_dataset()

release_pt_dataset = dataset[['Pitcher', 'AutoPitchType', 'RelHeight', 'RelSide']]

# ----- RELEASE POINT GRAPHS -----

# --- Release Point Data ---

release_pt_dataset = dataset[['Pitcher', 'AutoPitchType', 'RelHeight', 'RelSide']]

# --- Create Visualization Function ---

def release_pt_graph(name):
    selected_data = release_pt_dataset[release_pt_dataset['Pitcher'] == name]
    unique_pitches = np.unique(selected_data['AutoPitchType'])
    unique_colors = [pitch_color_key[unique_pitch] for unique_pitch in unique_pitches]
    colors_subset = []

    for pitch_type in selected_data['AutoPitchType']:
        colors_subset.append(pitch_color_key[pitch_type])
    
    today = date.today().strftime("%B %d, %Y")

    last_name, first_name = name.split(', ')
    title = f'{first_name} {last_name} Release Point ({today})'

    fig, ax = plt.subplots(num=title)

    scatter = ax.scatter(selected_data['RelSide'], selected_data['RelHeight'], c=colors_subset, alpha=0.7)
    ax.set(xlim=(-3, 3), ylim=(4.5, 6))
    ax.set_xlabel('Release Point X (in.)')
    ax.set_ylabel('Release Point Y (in.)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(c='white')
    ax.set_facecolor('#ebebeb')

    last_name, first_name = name.split(', ')
    ax.set_title(title)

    starting_height = 5.9

    for color, pitch in zip(unique_colors, unique_pitches):
        ax.add_patch(Ellipse((-2.5, starting_height+.015), .13, .05, color=color))
        ax.text(-2.25, starting_height, f'{pitch}')
        starting_height -= .1
    
    plt.show()

# --- Ask For Input ---

release_pt_graph(input('Choose pitcher > '))
