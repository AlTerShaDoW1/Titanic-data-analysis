import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')  # Make sure 'train.csv' is in your code folder

print(df.head())

print(df.info())
print(df.describe())

print(df.isnull().sum())

df['Age'].fillna(df['Age'].mean(), inplace=True)

sns.countplot(x='Survived', data=df)
plt.title('Survivors (1) vs Non-survivors (0)')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival By Gender')
plt.show()

print('Average Age of Survivors: ', df[df['Survived']==1]['Age'].mean())
print('Average Age of Non-Survivors: ', df[df['Survived']==0]['Age'].mean())

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()
