import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Read dataset from CSV
df = pd.read_csv('../clean_dataset.csv')
print(df.head(50))

# Data frame for analysis
analysis_df = df.iloc[:, :12]


# Plot histogram, dont say much
analysis_df.plot.hist()
plt.show()


# Correlation between all features
analysis_df.corr().style.background_gradient(cmap='coolwarm').set_precision(3)
plt.show()


# Scatter matrix, like we expected the correlations would be
scatter_matrix(analysis_df, alpha=0.3, figsize=(14, 8), diagonal='kde')
plt.show()


# Heatmap, the playing time do not interfere in the rest of the columns
sns.heatmap(analysis_df.corr(), linewidth=.5)
plt.show()


# Analyze champions influence
champions_dataset = pd.read_csv('../clean_dataset_with_champions.csv')


# Create a object with all the champions with the amount of times it appears in each elo
champion_eloCounter = {}
for row in champions_dataset.iterrows():
    row = row[1]
    if row.get('champion_1') in champion_eloCounter:
        if row.get('elo') in champion_eloCounter[row.get('champion_1')]:
            champion_eloCounter[row.get('champion_1')][row.get('elo')] += 1
        else:
            champion_eloCounter[row.get('champion_1')][row.get('elo')] = 1
    else:
        champion_eloCounter[row.get('champion_1')] = {row.get('elo'): 1}


# Create a reduced object of the collected champions
reduced_champion_obj = {}
for idx, champ in enumerate(champion_eloCounter):
    if idx == 20:
        break
    reduced_champion_obj[champ] = champion_eloCounter[champ]


# For each champion get the amount of times it appears in each elo, if it does not appear, then 0
bronze_bars = []
silver_bars = []
golden_bars = []
platinum_bars = []
diamond_bars = []
master_bars = []
challenger_bars = []

for champ in reduced_champion_obj:
    if 'Bronze' not in reduced_champion_obj[champ]:
        bronze_bars.append(0)
    else:
        bronze_bars.append(reduced_champion_obj[champ]['Bronze'])

    if 'Silver' not in reduced_champion_obj[champ]:
        silver_bars.append(0)
    else:
        silver_bars.append(reduced_champion_obj[champ]['Silver'])

    if 'Golden' not in reduced_champion_obj[champ]:
        golden_bars.append(0)
    else:
        golden_bars.append(reduced_champion_obj[champ]['Golden'])

    if 'Platinum' not in reduced_champion_obj[champ]:
        platinum_bars.append(0)
    else:
        platinum_bars.append(reduced_champion_obj[champ]['Platinum'])

    if 'Diamond' not in reduced_champion_obj[champ]:
        diamond_bars.append(0)
    else:
        diamond_bars.append(reduced_champion_obj[champ]['Diamond'])

    if 'Master' not in reduced_champion_obj[champ]:
        master_bars.append(0)
    else:
        master_bars.append(reduced_champion_obj[champ]['Master'])

    if 'Challenger' not in reduced_champion_obj[champ]:
        challenger_bars.append(0)
    else:
        challenger_bars.append(reduced_champion_obj[champ]['Challenger'])


# Create figure
fig, ax = plt.subplots(figsize=(50, 50))

ind = np.arange(len(reduced_champion_obj))
width = .2


# Create the bars
bronze_bar = ax.bar(ind, bronze_bars, width)
silver_bar = ax.bar(ind + width, silver_bars, width)
golden_bar = ax.bar(ind + width * 2, golden_bars, width)
platinum_bar = ax.bar(ind + width * 3, platinum_bars, width)
diamond_bar = ax.bar(ind + width * 4, diamond_bars, width)
master_bar = ax.bar(ind + width * 5, master_bars, width)
challenger_bar = ax.bar(ind + width * 6, challenger_bars, width)


# Set the location and label (champion)
ax.set_xticks(ind + width)
ax.set_xticklabels((reduced_champion_obj.keys()))
ax.tick_params(axis='both', labelsize=56, which='major', rotation=45)
ax.tick_params(axis='both', labelsize=56, which='major', rotation=45)


# Add the subtitles
ax.legend((bronze_bar[0], silver_bar[0], golden_bar[0], platinum_bar[0], diamond_bar[0], master_bar[0], challenger_bar[0]), ('Bronze', 'Prata', 'Ouro', 'Platina', 'Diamante', 'Master', 'Desafiante'), prop=dict(size=74))
ax.autoscale_view()


# Show the image
plt.show()
