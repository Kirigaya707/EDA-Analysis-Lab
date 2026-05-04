# %% [markdown]
# # Exploratory Data Analysis (EDA) on the Titanic Dataset
# 
# **Objective**: Perform a thorough EDA to uncover patterns, relationships, and insights about the passengers and their survival.
# 
# **Tools**: pandas, numpy, matplotlib, seaborn, plotly

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
if 'ipykernel' in sys.modules:
    from IPython.display import display
else:
    display = print

# Set style for static plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## 2. Load Dataset
# We'll load Titanic data directly from Seaborn's built-in datasets.

# %%
# Load dataset
df = sns.load_dataset('titanic')

# Create a copy for safe manipulation
data = df.copy()

# Display first rows
print("First 5 rows of the dataset:")
display(data.head())

# %% [markdown]
# ## 3. Initial Data Inspection

# %%
# Dataset shape
print(f"\nDataset shape: {data.shape}")

# Column names and data types
print("\nData types and non-null counts:")
print(data.info())

# Summary statistics for numerical columns
print("\nDescriptive statistics (numerical):")
display(data.describe())

# Summary statistics for categorical columns
print("\nDescriptive statistics (categorical):")
display(data.describe(include=['object', 'category']))

# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# %% [markdown]
# ## 4. Data Cleaning & Feature Engineering

# %%
# --- Handle missing values ---

# 'age' missing: fill with median age grouped by passenger class and sex
age_median = data.groupby(['pclass', 'sex'])['age'].transform('median')
data['age'].fillna(age_median, inplace=True)

# 'embarked' missing (only 2): fill with mode
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)

# 'deck' (from 'cabin') – too many missing; we'll extract first letter of cabin instead
data['deck'] = data['cabin'].str[0]  # first letter (A, B, C, ...)

# 'embark_town' missing = same as 'embarked'
data['embark_town'].fillna(data['embarked'].map({ 'S':'Southampton', 'C':'Cherbourg', 'Q':'Queenstown' }), inplace=True)

# Drop 'alive' and 'who' as they are redundant (alive = survived mapping, who = child/man/woman)
data.drop(['alive', 'who'], axis=1, inplace=True)

# --- Feature engineering ---
# Family size = siblings/spouse + parents/children + self
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Is alone? (family_size == 1)
data['is_alone'] = (data['family_size'] == 1).astype(int)

# Extract title from name – raw string (r'...') to avoid SyntaxWarning
data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Group rare titles
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
data['title'] = data['title'].replace(rare_titles, 'Rare')
data['title'] = data['title'].replace(['Mlle', 'Ms'], 'Miss')
data['title'] = data['title'].replace('Mme', 'Mrs')

# Verify no missing values remain (except 'deck' which we keep as is)
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Show cleaned data preview
display(data.head())

# %% [markdown]
# ## 5. Univariate Analysis

# %%
# --- Numerical Features ---

num_cols = ['age', 'fare', 'family_size', 'sibsp', 'parch']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(data[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.show()

# Boxplots to detect outliers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(y=data[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# --- Categorical Features ---

cat_cols = ['sex', 'pclass', 'embarked', 'deck', 'title', 'is_alone', 'survived']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=data, x=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Count of {col}')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Bivariate Analysis (Survival vs. Features)

# %%
# Survival rate by categorical features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

cat_surv = ['sex', 'pclass', 'embarked', 'deck', 'title', 'is_alone']
for i, col in enumerate(cat_surv):
    survival_rate = data.groupby(col)['survived'].mean().sort_values(ascending=False)
    sns.barplot(x=survival_rate.index, y=survival_rate.values, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Survival Rate by {col}')
    axes[i].set_ylabel('Survival Rate')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Numerical features vs. survival (boxplots)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

num_surv = ['age', 'fare', 'family_size', 'sibsp']
for i, col in enumerate(num_surv):
    sns.boxplot(data=data, x='survived', y=col, ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'{col} vs. Survival')
plt.tight_layout()
plt.show()

# Pairplot for numerical features colored by survival
sns.pairplot(data[num_surv + ['survived']], hue='survived', diag_kind='kde', palette='husl')
plt.suptitle('Pairplot of Numerical Features Colored by Survival', y=1.02)
plt.show()

# %% [markdown]
# ## 7. Multivariate Analysis

# %%
# Correlation heatmap (numerical features only, excluding target)
corr = data[['age', 'fare', 'sibsp', 'parch', 'family_size', 'survived']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Survival by Pclass and Sex (grouped bar plot)
pclass_sex_surv = data.groupby(['pclass', 'sex'])['survived'].mean().unstack()
pclass_sex_surv.plot(kind='bar', figsize=(10, 6), colormap='coolwarm')
plt.title('Survival Rate by Passenger Class and Sex')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.xticks(rotation=0)
plt.legend(title='Sex')
plt.show()

# FacetGrid: Age distribution by Pclass and Sex, split by survival
g = sns.FacetGrid(data, col='survived', row='sex', hue='pclass', height=4, aspect=1.5)
g.map(sns.kdeplot, 'age', fill=True, alpha=0.6)
g.add_legend(title='Pclass')
g.set_axis_labels('Age', 'Density')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Age Distribution: Survived vs. Not Survived, by Sex and Pclass')
plt.show()

# %% [markdown]
# ## 8. Interactive Visualizations with Plotly

# %%
# Interactive scatter plot: Age vs. Fare, colored by Survival, sized by Family Size
fig = px.scatter(data, x='age', y='fare', color='survived', 
                 size='family_size', hover_data=['sex', 'pclass', 'title'],
                 title='Interactive: Age vs. Fare (Color=Survival, Size=Family Size)',
                 labels={'survived':'Survived', 'family_size':'Family Size'},
                 color_continuous_scale='Viridis')
fig.show()

# Interactive 3D plot: Age, Fare, Pclass, colored by Survival
fig = px.scatter_3d(data, x='age', y='fare', z='pclass', color='survived',
                    size='family_size', hover_data=['sex', 'title'],
                    title='3D Plot: Age, Fare, and Passenger Class (Color=Survival)',
                    labels={'pclass':'Passenger Class'})
fig.show()

# Interactive stacked bar chart: Survival by Embarked and Sex
surv_emb_sex = data.groupby(['embarked', 'sex'])['survived'].value_counts().unstack().fillna(0)
surv_emb_sex = surv_emb_sex.rename(columns={0:'Died', 1:'Survived'})
fig = px.bar(surv_emb_sex, x=surv_emb_sex.index, y=['Died', 'Survived'], 
             title='Survival Count by Embarkation Port and Sex',
             labels={'value':'Count', 'variable':'Status', 'embarked':'Embarkation Port'},
             barmode='group', color_discrete_sequence=['coral', 'lightgreen'])
fig.show()

# Interactive histogram of age with slider for Pclass
fig = px.histogram(data, x='age', color='survived', facet_col='pclass',
                   title='Age Distribution by Passenger Class and Survival',
                   labels={'age':'Age', 'count':'Number of Passengers'},
                   barmode='overlay', nbins=30)
fig.show()

# %% [markdown]
# ## 9. Key Insights & Summary
# 
# - **Survival rate** was significantly higher for:
#   - Women (approx. 74%) than men (19%).
#   - First‑class passengers (63%) compared to third‑class (24%).
#   - Children and passengers with smaller families (family_size = 1–3).
# - **Age**: Younger children (under 10) had higher survival chance.
# - **Fare**: Higher fares correlated with better survival (wealthier passengers).
# - **Embarkation**: Passengers from Cherbourg (C) had higher survival rate, possibly due to a higher proportion of first‑class passengers.
# - **Missing data**: Age was imputed using median values per class/sex; Cabin/Deck had too many missing values to be reliable.
# - **Interactive plots** provided deeper exploration of relationships.

# %%
# Optional: Save cleaned dataset
# data.to_csv('titanic_cleaned.csv', index=False)