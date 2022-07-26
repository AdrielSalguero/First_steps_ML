from tkinter.tix import Tree
from matplotlib.colors import cnames
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

train_df = pd.read_csv('C:/Users/Usuario/Documents/Codes/ML-Fundamentos/data/titanic-train.csv')
test_df = pd.read_csv('C:/Users/Usuario/Documents/Codes/ML-Fundamentos/data/titanic-test.csv')

#print(train_df.head(10))

#print(train_df.info())

label_encoder = preprocessing.LabelEncoder()

encoder_sex = label_encoder.fit_transform(train_df['Sex'])
train_df['Sex'] = encoder_sex
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')
print('Primer DF c/ transformacon en sex, age y embarked' , train_df.head(10))
#Predictores 
#Creo los predictores eliminando columnas que no suman valores
train_predictors = train_df.drop(['PassengerId', 'Survived','Name','Ticket','Cabin'], axis = 1)

print('DF train predictors \n', train_predictors.head(10))
#busco las columnas categoricas y uso un factor de seguridad que no tenga mas de 10 valores unicos
categorical_cols = [cname for cname in train_predictors.columns if 
                        train_predictors[cname].nunique()<10 and
                        train_predictors[cname].dtype == 'object']
print('Categorical cols \n', categorical_cols)

#sepero las columnas numericas
numerical_cols = [cname for cname in train_predictors.columns if
                    train_predictors[cname].dtype in ['int64', 'float64']]
print('Numerical cols \n', numerical_cols)
my_cols = categorical_cols+ numerical_cols

print('My cols \n', train_predictors.head(10))


train_predictors = train_predictors[my_cols]

print('DF train predictors con my cols \n', train_predictors.head(10))


dummy_encoded_train_predictors = pd.get_dummies(train_predictors)

print('Dummies \n', dummy_encoded_train_predictors.head(10))

train_df['Pclass'].value_counts()
print(train_df.head(10))

y_target = train_df['Survived'].values
x_features = dummy_encoded_train_predictors.values

x_train,x_validation, y_train, y_validation = train_test_split(x_features, y_target, test_size= .25)
tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(x_features, y_target)
tree_one_acuracy = round(tree_one.score(x_features, y_target),4)
print('Accuary ',tree_one_acuracy)


#Visualizacion

print('-------------Visualizacion-------')

from io import StringIO
#
from IPython.display import Image, display
#Importamos para poder generar cada uno de los caminos
import pydotplus 

#trabajamos con archivos externos
out = StringIO()
#permite interactuar y crear imagenes
tree.export_graphviz(tree_one, out_file = out)

#para generar las salidas traemos con pydot

graph= pydotplus.graph_from_dot_data(out.getvalue())

graph.write_png('Titanic.png')
