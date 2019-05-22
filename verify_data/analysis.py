import pandas as pd
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


"""
Conclusion:
Maybe drop the games column and/or the pentakills column since they are to much correlated with the rest.
Let's test with all, without pentakills, without games and without both
"""
