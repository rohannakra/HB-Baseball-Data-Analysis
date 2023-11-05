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
from tabulate import tabulate
from datetime import date

dataset = pd.read_csv('C:/Users/rohan/Documents/AI/Projects/HB Baseball Analysis/data/data.csv')

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

pitch_chart_dataset = dataset[['Pitcher', 'AutoPitchType', 'RelSpeed', 'PitchCall',
                               'InducedVertBreak', 'HorzBreak', 'RelHeight', 'RelSide', 'SpinRate', 'Extension']]

# ----- CREATE CHART FUNCTION -----


def pitch_chart(name):
    selected_data = pitch_chart_dataset[pitch_chart_dataset['Pitcher'] == name]
    unique_pitches = np.unique(selected_data['AutoPitchType'])

    columns = unique_pitches
    
    # Setting variables for each row.
    max_velos = []
    avg_velos = []
    strikes_db_pitches = []
    strike_percentage = []
    whiff_percentage = []
    horizontal_break = []
    vertical_break = []
    release_height = []
    release_side = []
    spin_rate = []
    extension = []

    for pitch_type in unique_pitches:
        selected_pitch_data = selected_data[selected_data['AutoPitchType'] == pitch_type]

        velos = list(selected_pitch_data['RelSpeed'])

        max_velos.append(f'{max(velos):.1f}')
        avg_velos.append(f'{sum(velos)/len(velos):.1f}')

        pitch_calls = selected_pitch_data['PitchCall']
        strikes, total_pitches = (0, 0)

        for call in pitch_calls:
            if call != 'BallCalled' and call != 'InPlay':
                strikes += 1
            total_pitches += 1

        strikes_db_pitches.append(f'{strikes}/{total_pitches}')
        strike_percentage.append(f'{strikes/total_pitches * 100:.1f}%')
        
        swing_and_miss, total_swings = (0, 0)

        for call in pitch_calls:
            if call == 'StrikeSwinging':
                swing_and_miss += 1
                total_swings += 1
            elif call == 'InPlay' or call == 'FoulBall':
                total_swings += 1

        try:
            whiff_percentage.append(f'{swing_and_miss/total_swings*100:.1f}%')
        except ZeroDivisionError:
            whiff_percentage.append(f'--')

        horiz_breaks = list(selected_pitch_data['HorzBreak'])
        horizontal_break.append(f'{sum(horiz_breaks)/len(horiz_breaks):.1f}')

        vert_breaks = list(selected_pitch_data['InducedVertBreak'])
        vertical_break.append(f'{sum(vert_breaks)/len(vert_breaks):.1f}')

        release_heights = list(selected_pitch_data['RelHeight'])
        release_height.append(f'{sum(release_heights)/len(release_heights):.1f}')

        release_sides = list(selected_pitch_data['RelSide'])
        release_side.append(f'{sum(release_sides)/len(release_sides):.1f}')

        spin_rates = list(selected_pitch_data['SpinRate'])
        spin_rate.append(f'{sum(spin_rates)/len(spin_rates):.1f}')

        extensions = list(selected_pitch_data['Extension'])
        extension.append(f'{sum(extensions)/len(extensions):.1f}')

    chart = pd.DataFrame(
        data=[max_velos, avg_velos, strikes_db_pitches, strike_percentage, whiff_percentage, horizontal_break, vertical_break, release_height, release_side, spin_rate, extension], 
        columns=[unique_pitches], 
        index=['Maximum Velocity', 'Average Velocity', '# of Strikes/Pitches', 'Strike %', 'Whiff %', 'Vertical Movement', 'Horizontal Movement', 'Release Height', 'Release Side', 'Spin Rate (RPM)', 'Extension']
    )

    today = date.today().strftime("%B %d, %Y")
    
    last_name, first_name = name.split(', ')
    print(f'\n\n\t\t\t{first_name.upper()} {last_name.upper()} PITCH CHART ({today})')
    print(tabulate(chart, headers=unique_pitches, tablefmt='fancy_grid'))


pitch_chart(input('Choose pitcher > '))
