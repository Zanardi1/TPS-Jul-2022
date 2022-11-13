# https://www.kaggle.com/code/thedevastator/bruteforce-clustering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import PowerTransformer

sns.set_style('darkgrid')

data = pd.read_csv('data.csv')
print(data.head())
print(data.shape)
print(data.info())
print(data.describe().to_string())

submission = pd.read_csv('sample_submission.csv')

mask = np.triu(np.ones_like(data.corr(), dtype='bool'))
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), mask=mask, annot=True, fmt='.2f')
plt.show()

data = data.drop(columns='id')
data = data[
    ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28']]
cols = list(data.columns)

X_scaled = PowerTransformer().fit_transform(data)
X_scaled = pd.DataFrame(X_scaled, columns=cols)

pca = PCA(random_state=10, whiten=True)
X_pca = pca.fit_transform(X_scaled)
PCA_df = pd.DataFrame({'PCA_1': X_pca[:, 0], 'PCA_2': X_pca[:, 1]})
plt.figure(figsize=(14, 14))
sns.scatterplot(data=PCA_df, x='PCA_1', y='PCA_2', s=3)
plt.show()

gmm = BayesianGaussianMixture(n_components=7, n_init=5, covariance_type='full')
preds = gmm.fit_predict(X_scaled)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_scaled)
df = pd.DataFrame({'x': reduced_data[:, 0], 'y': reduced_data[:, 1], 'clusters': preds})
plt.figure(figsize=(20, 10))
sns.scatterplot(x=df['x'], y=df['y'], hue=df['clusters'])
plt.show()

submission['Predicted'] = preds
submission.to_csv('submission.csv', index=False)
