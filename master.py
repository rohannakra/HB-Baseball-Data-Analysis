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

pitch_color_key = {    # for the legend on graphs.
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

def get_dataset():    # runs recursively until a valid file path is given.
    os.system('cls')
    sleep(2)
    dataset_path = input('Dataset path > ')

    try:
        return pd.read_csv(dataset_path)
    except FileNotFoundError:
        print('File not found. Please try again.')
        sleep(2)
        get_dataset()


dataset = get_dataset()

# Dataset for each graph/table.
pitch_break_dataset = dataset[['Pitcher', 'AutoPitchType', 'InducedVertBreak', 'HorzBreak']]
pitch_break_dataset.dropna(inplace=True)    # drop na instances to ensure no errors.

pitch_chart_dataset = dataset[[
    'Pitcher', 'AutoPitchType', 'RelSpeed', 'PitchCall', 'InducedVertBreak', 
    'HorzBreak', 'RelHeight', 'RelSide', 'SpinRate', 'Extension'
]]
pitch_chart_dataset.dropna(inplace=True)

release_pt_dataset = dataset[['Pitcher', 'AutoPitchType', 'RelHeight', 'RelSide']]
release_pt_dataset.dropna(inplace=True)

# ----- PITCH BREAK GRAPH -----

def pitch_break_graph(name, ax):

    # Initialize variables.
    selected_data = pitch_break_dataset[pitch_break_dataset['Pitcher'] == name]
    unique_pitches = np.unique(selected_data['AutoPitchType'])
    unique_colors = [pitch_color_key[unique_pitch]
                     for unique_pitch in unique_pitches]
    colors_subset = []

    # Update pitch key for only the set pitches that the specified pitcher throws.
    for pitch_type in selected_data['AutoPitchType']:
        colors_subset.append(pitch_color_key[pitch_type])

    today = date.today().strftime("%B %d, %Y")
    last_name, first_name = name.split(', ')

    # Set up graph.
    ax.scatter(selected_data['HorzBreak'], selected_data['InducedVertBreak'], c=colors_subset, alpha=0.7)
    ax.set_xlabel('Horizontal Break (in.)')
    ax.set_ylabel('Induced Vertical Break (in.)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(c='white')
    ax.set_facecolor('#ebebeb')
    ax.set(xlim=(-25, 25), ylim=(-27, 27))
    ax.set_title('Pitch Break')

    starting_height = -12

    ax.plot([-25, 25], [0, 0], c='black')
    ax.plot([0, 0], [-40, 40], c='black')

    # Create legend.
    for color, pitch in zip(unique_colors, unique_pitches):
        ax.add_patch(Ellipse((11, starting_height+.6), 1, 1.5, color=color))
        ax.text(12, starting_height, f'{pitch}')
        starting_height -= 3

# ----- RELEASE POINT GRAPH -----

def release_pt_graph(name, ax):

    # Initialize variables.
    selected_data = release_pt_dataset[release_pt_dataset['Pitcher'] == name]
    unique_pitches = np.unique(selected_data['AutoPitchType'])
    unique_colors = [pitch_color_key[unique_pitch] for unique_pitch in unique_pitches]
    colors_subset = []

    # Update pitch key for only the set pitches that the specified pitcher throws.
    for pitch_type in selected_data['AutoPitchType']:
        colors_subset.append(pitch_color_key[pitch_type])
    
    today = date.today().strftime("%B %d, %Y")
    last_name, first_name = name.split(', ')

    # Set up graph.
    scatter = ax.scatter(selected_data['RelSide'], selected_data['RelHeight'], c=colors_subset, alpha=0.7)
    ax.set(xlim=(-3, 5), ylim=(3, 7.3))
    ax.set_xlabel('Release Point X (in.)')
    ax.set_ylabel('Release Point Y (in.)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(c='white')
    ax.set_facecolor('#ebebeb')

    last_name, first_name = name.split(', ')
    ax.set_title('Release Point')

    starting_height = 5.9
    starting_height_circle = 5.95

    # Create Legend.
    for color, pitch in zip(unique_colors, unique_pitches):
        ax.add_patch(Ellipse((-2.5, starting_height_circle), .15, .1, color=color))
        ax.text(-2.25, starting_height, f'{pitch}')
        starting_height -= .3
        starting_height_circle -= .3

# ----- PITCH TABLE -----

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

    # Looping over pitch by pitch... adding to the aforementioned lists as needed.
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

    # Summarize in DataFrame.
    chart = pd.DataFrame(
        data=[
            max_velos, avg_velos, strikes_db_pitches, strike_percentage, whiff_percentage, 
            horizontal_break, vertical_break, release_height, release_side, spin_rate, extension
        ], 
        columns=[unique_pitches], 
        index=[
            'Maximum Velocity', 'Average Velocity', '# of Strikes/Pitches', 
            'Strike %', 'Whiff %', 'Vertical Movement', 'Horizontal Movement', 
            'Release Height', 'Release Side', 'Spin Rate (RPM)', 'Extension'
        ]
    )

    # Convert to matplotlib table.
    table = plt.table(
        cellText=chart.values, 
        colLabels=tuple(value[0] for value in chart.columns), 
        rowLabels=[
            'Maximum Velocity', 'Average Velocity', '# of Strikes/Pitches', 'Strike %', 
            'Whiff %', 'Vertical Movement', 'Horizontal Movement', 'Release Height', 
            'Release Side', 'Spin Rate (RPM)', 'Extension'
        ],
        loc='top'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(.7, .6)

    return table

# ----- START MASTER SCRIPT -----

def start():

    print('\n')
    print('      --- Choose Pitcher --- \n')

    for pitcher in np.unique(dataset['Pitcher']):
        last_name, first_name = pitcher.split(', ')
        print(f'\t - {first_name} {last_name}')

    pitcher = input('\nChoose pitcher > ')

    def run(name):
        today = date.today().strftime("%B %d, %Y")
        last_name, first_name = pitcher.split(', ')
        title = f'{first_name} {last_name} Data Analysis ({today})'

        fig, axs = plt.subplots(2, 2, num=title)
        fig.suptitle(title, fontsize=16)

        pitch_break_graph(pitcher, axs[0, 0])
        release_pt_graph(pitcher, axs[1, 0])
        axs[1, 1].axis('off')
        axs[1, 1] = pitch_chart(pitcher)

        axs[0, 1].axis('off')


        plt.show()
    
    if pitcher == 'all':
        for pitcher in np.unique(dataset['Pitcher']):
            run(pitcher)
    else:
        run(pitcher)

start()
