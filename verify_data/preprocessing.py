import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from verify_data.label_encoder import LabelEncoder

K = 1000
M = 1000000


# Read the csv and load to variable df
df = pd.read_csv('../dataset.csv')
df = df.dropna()


# Drop the unrankeds, can not use the lines although they are a significant number, there is no other way
df = df[df.elo != 'Unranked']


# Remove all the string parts and convert to floats
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


df_without_champions = df.drop(['elo', 'champion_1', 'champion_2', 'champion_3'], axis=1)
df_without_champions_columns = df_without_champions.columns
df_without_champions = df_without_champions.apply(convert_to_float)


# Scale the values
scale = StandardScaler()
scaled_df = pd.DataFrame(scale.fit_transform(df_without_champions), columns=list(df_without_champions_columns))


# Encode the champions
one_hot_encoder = OneHotEncoder(categories='auto')
label_encoder = LabelEncoder()

champion_3 = df[['champion_3']]
champion_2 = df[['champion_2']]
champion_1 = df[['champion_1']]

champion_3 = pd.DataFrame(label_encoder.fit_transform(champion_3))
champion_2 = pd.DataFrame(label_encoder.fit_transform(champion_2))
champion_1 = pd.DataFrame(label_encoder.fit_transform(champion_1))

champion_3 = pd.DataFrame(one_hot_encoder.fit_transform(champion_3).toarray())
size_3 = len(champion_3.columns)

champion_2 = pd.DataFrame(one_hot_encoder.fit_transform(champion_2).toarray())
size_2 = len(champion_2.columns)
champion_2.columns = list(range(size_3, size_3 + size_2))

champion_1 = pd.DataFrame(one_hot_encoder.fit_transform(champion_1).toarray())
size_1 = len(champion_1.columns)
champion_1.columns = list(range(size_3 + size_2, size_3 + size_2 + size_1))


# Remove the excess of info from the target, like Bronze III, the III is not needed
def remove_excess(column):
    values = []
    for idx, line in column.iterrows():
        line = list(line)[0]
        if str(line).lower().startswith('bronze') or str(line).lower().endswith('bronze'):
            values.append('Bronze')
        elif str(line).lower().startswith('silver') or str(line).lower().endswith('silver'):
            values.append('Silver')
        elif str(line).lower().startswith('gold') or str(line).lower().endswith('gold'):
            values.append('Gold')
        elif str(line).lower().startswith('platinum') or str(line).lower().endswith('platinum'):
            values.append('Platinum')
        elif str(line).lower().startswith('diamond') or str(line).lower().endswith('diamond'):
            values.append('Diamond')
        elif str(line).lower().startswith('master') or str(line).lower().endswith('master'):
            values.append('Master')
        elif str(line).lower().startswith('challenger') or str(line).lower().endswith('challenger'):
            values.append('Challenger')

    return values


target_column = pd.DataFrame(remove_excess(df[['elo']]), columns=['elo'])


# Concat the champions data frames with the scaled values
df = pd.concat([target_column, scaled_df, champion_3, champion_2, champion_1], axis=1).dropna()
print(df.head(50))


# # Save the dataframe
df.to_csv('../clean_dataset.csv', index=False)
