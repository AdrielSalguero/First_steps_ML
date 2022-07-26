#importamos Skitlearn

from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#dataset por defecto en SKLEARN
iris = datasets.load_iris()

#dividimos nuestros datos de entrenamientos y las etiquetas
x_iris = iris.data
#
y_iris = iris.target

x = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.DataFrame(iris.target, columns = ['Target'])

#Analizamos que trae nuestro DF
print(x.head(10), x.columns)
plt.scatter(x['petal length (cm)'], x['petal width (cm)'], c= 'Blue')
plt.xlabel('Petal Lenghth', fontsize = 10)
plt.ylabel('Petal Width', fontsize = 10)
plt.show()

plt.scatter(x['sepal length (cm)'], x['sepal width (cm)'], c= 'Green')
plt.xlabel('Sepal Length', fontsize = 10)
plt.ylabel('Sepal Width', fontsize = 10)
plt.show()

#Construccion de y evaluacion de modelo K-means
#n_clusters es la similitud a K lo que permite generar los centroides
#
model = KMeans(n_clusters= 3 ,max_iter= 1000)
model.fit(x)
y_labels = model.labels_
y_kmeans = model.predict(x)
print('Predicciones: ', y_kmeans)

#vemos si los 2 k son suficientes

accuracy = metrics.adjusted_rand_score(y_iris, y_kmeans)
print(accuracy)

#confirmacion visual de los clusters
plt.scatter(x['petal length (cm)'], x['petal width (cm)'], c=y_kmeans, s=20)
plt.xlabel('Petal Lenghth', fontsize = 10)
plt.ylabel('Petal Width', fontsize = 10)
plt.show()


plt.scatter(x['sepal length (cm)'], x['sepal width (cm)'], c=y_kmeans)
plt.xlabel('Sepal Length', fontsize = 10)
plt.ylabel('Sepal Width', fontsize = 10)
plt.show()
