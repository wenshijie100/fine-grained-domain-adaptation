import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import my_init as myinit
import my_normlization as mynormlization
sns.set(style="white", color_codes=True)

#加载iris数据集
#from sklearn.datasets import load_iris

X_train1,Y_train1=myinit.quick_77G_dataset()
X_train1=X_train1[::10]
X_train1=X_train1[:,0:10]
Y_train1=Y_train1[::10]
X_train,Y_train=myinit.quick_own_dataset()
Y_train=Y_train.astype(int)
X_train=X_train[:,0:10]

X_train=np.concatenate((X_train,X_train1),axis=0)
Y_train=np.concatenate((Y_train,Y_train1),axis=0)
X_train=mynormlization.normlization3(X_train)
#print(Y_train)

#iris_data = load_iris()
feature_names=['x_'+str(i) for i in range(X_train.shape[1])]
#print("show data:",iris_data['data'][0],iris_data['feature_names'],iris_data['target'])
iris = pd.DataFrame(X_train, columns=feature_names)
iris = pd.merge(iris, pd.DataFrame(Y_train, columns=['species']), left_index=True, right_index=True)
'''
print("head",iris.head())
# look at an individual feature in Seaborn through a boxplot
#pd.plotting.parallel_coordinates(iris, 'species')
sns.boxplot(x='species', y='x_3', data=iris)
# kdeplot looking at univariate relations
# creates and visualizes a kernel density estimate of the underlying feature

sns.FacetGrid(iris, hue='species',size=6) \
   .map(sns.kdeplot, 'x_3') \
    .add_legend()
# A violin plot combines the benefits of the boxplot and kdeplot 
# Denser regions of the data are fatter, and sparser thiner in a violin plot

sns.violinplot(x='species', y='x_3', data=iris, size=6)

# use seaborn's FacetGrid to color the scatterplot by species

sns.FacetGrid(iris, hue="species", size=5) \
    .map(plt.scatter, "x_3", "x_4") \
    .add_legend()
# pairplot shows the bivariate relation between each pair of features
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other two across all feature combinations
# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde

sns.pairplot(iris, hue='species', size=3, diag_kind='kde')


# Andrews Curves involve using attributes of samples as coefficients for Fourier series and then plotting these

#pd.plotting.andrews_curves(iris, 'species')

# Parallel coordinates plots each feature on a separate column & then draws lines connecting the features for each data sample

#pd.plotting.parallel_coordinates(iris, 'species')

# radviz  puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted by the relative value for that feature

colors = {'red', 'blue', 'green', 'black'}
pd.plotting.radviz(iris, 'species',colormap = 'brg')
'''
colors = {'red', 'blue', 'green', 'black'}
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(iris)
print("S2:",X_train.shape,X_tsne.shape)

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(Y_train[i]), color=plt.cm.Set1(Y_train[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.savefig('my_plot.jpg')
plt.show()