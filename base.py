# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the data
df = pd.read_csv('train.csv')  # Make sure 'train.csv' is in your code folder

# 3. Take a look at the data
print(df.head())

# 4. Get a summary
print(df.info())
print(df.describe())

# 5. Check for missing values
print(df.isnull().sum())

# 6. Fill missing Age values with the mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 7. Data analysis: Survival Count
sns.countplot(x='Survived', data=df)
plt.title('Survivors (1) vs Non-survivors (0)')
plt.show()

# 8. Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival By Gender')
plt.show()

# 9. Average age of survivors vs non-survivors
print('Average Age of Survivors: ', df[df['Survived']==1]['Age'].mean())
print('Average Age of Non-Survivors: ', df[df['Survived']==0]['Age'].mean())

# 10. Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()
