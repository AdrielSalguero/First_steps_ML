import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics

wine = datasets.load_wine()

#normalizamos los datos para poder comparar entre ellos mismos en un misma escala

scaler = StandardScaler()
scaler.fit(wine.data)
norm_wines = scaler.transform(wine.data)
y_wines = wine.target

x = pd.DataFrame(norm_wines, columns = wine.feature_names)
y = pd.DataFrame(wine.target, columns = ['target'])

print(x.head(10), x.columns)


corr_df = x.corr(method = 'pearson')

plt.figure(figsize= (20,18))
sns.heatmap(corr_df, annot= True)
plt.show()

#A partir de la correlacion existente entre los datos utilizamos las columnas total_phenols and flavanoids
#ya que son las que representan una correlacion cercan a 1 

plt.scatter(x['total_phenols'],x['flavanoids'], c= 'Red', s= 25)
plt.xlabel('Total phenols', fontsize= 10)
plt.ylabel('Flavanoids', fontsize= 10)
plt.show()

#Construccion del modelo

model = KMeans(n_clusters = 3, max_iter= 1000)
x_new = x[['total_phenols', 'flavanoids','od280/od315_of_diluted_wines']]
model.fit(x_new) 
y_labels = model.labels_
y_kmeans = model.predict(x_new)

plt.scatter(x['total_phenols'],x['flavanoids'], c= y_kmeans, s= 25)
plt.xlabel('Total phenols', fontsize= 10)
plt.ylabel('Flavanoids', fontsize= 10)
plt.show()

accuaracy= metrics.adjusted_rand_score(y_wines, y_kmeans)
print(accuaracy)
