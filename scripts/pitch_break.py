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

dataset = pd.read_csv('./data/data.csv')

pitch_color_key = {
    'Changeup': 'blue',
    'Curveball': 'lime',
    'Cutter': 'indigo',
    'Four-Seam': 'red',
    'Other': 'gray',
    'Sinker': 'darkorange',
    'Slider': 'magenta',
    'Splitter': 'cyan'
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

pitch_break_dataset = dataset[[
    'Pitcher', 'AutoPitchType', 'InducedVertBreak', 'HorzBreak']]

# ----- PITCH BREAK GRAPHS -----

# --- Create Visualization Function ---


def pitch_break_graph(name):
    selected_data = pitch_break_dataset[pitch_break_dataset['Pitcher'] == name]
    unique_pitches = np.unique(selected_data['AutoPitchType'])
    unique_colors = [pitch_color_key[unique_pitch]
                     for unique_pitch in unique_pitches]
    colors_subset = []

    for pitch_type in selected_data['AutoPitchType']:
        colors_subset.append(pitch_color_key[pitch_type])

    today = date.today().strftime("%B %d, %Y")

    last_name, first_name = name.split(', ')
    title = f'{first_name} {last_name} Pitch Break ({today})'

    fig, ax = plt.subplots(num=title)

    scatter = ax.scatter(selected_data['HorzBreak'], selected_data['InducedVertBreak'], c=colors_subset, alpha=0.7)
    ax.set_xlabel('Horizontal Break (in.)')
    ax.set_ylabel('Induced Vertical Break (in.)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(c='white')
    ax.set_facecolor('#ebebeb')
    ax.set(xlim=(-25, 25), ylim=(-27, 27))
    ax.set_title(title)

    starting_height = -7

    ax.plot([-25, 25], [0, 0], c='black')
    ax.plot([0, 0], [-40, 40], c='black')

    for color, pitch in zip(unique_colors, unique_pitches):
        ax.add_patch(Ellipse((11, starting_height+.75), .8, 1.5, color=color))
        ax.text(12, starting_height, f'{pitch}')
        starting_height -= 3

    plt.show()

# --- Ask For Input ---


pitch_break_graph(input('Choose pitcher > '))
