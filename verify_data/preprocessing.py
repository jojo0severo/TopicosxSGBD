import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

K = 1000
M = 1000000


df = pd.read_csv('../dataset.csv')
scale = StandardScaler()
encoder = OneHotEncoder()


# Drop the unrankeds, can not use the lines although they are a significant number, there is no other way
df = df[df.elo != 'Unranked']


# Remove the target column from the dataset and assign it to a variable
targets = df[['elo']]
df = df.drop(['elo'], axis=1)


# Scale the values
def convert_to_float(column):
    values = []
    for line in column:
        if str(line).lower().__contains__('m'):
            line = str(line).lower().replace('m', '').replace('m', '').replace('m', '')
            values.append(float(line) * M)
        elif str(line).lower().__contains__('k'):
            line = str(line).lower().replace('k', '').replace('k', '').replace('k', '')
            values.append(float(line) * K)
        else:
            values.append(float(line))

    return values


df_without_champions = df.drop(['champion_1', 'champion_2', 'champion_3'], axis=1)
df_without_champions = df_without_champions.apply(convert_to_float)
scaled_df = pd.DataFrame(scale.fit_transform(df_without_champions))
df = pd.concat([df[['champion_1', 'champion_2', 'champion_3']], scaled_df], axis=1)


# Encode the champions
df = df.dropna()
ordinal_df = pd.DataFrame(encoder.fit_transform(df[['champion_3', 'champion_2', 'champion_1']]).toarray())
df = pd.concat([df.drop(['champion_3', 'champion_2', 'champion_1'], axis=1), ordinal_df], axis=1)


# Save the dataframe
df = pd.concat([df, targets], axis=1)
df = df.dropna()
df.to_csv('../clean_dataset.csv')
