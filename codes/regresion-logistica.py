import pandas as pd
from sklearn import metrics
#
from sklearn.model_selection import train_test_split
#Para clasificar
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv('C:/Users/Usuario/Documents/Codes/ML-Fundamentos/data/diabetes-ml.csv')
print(dataset.head(10))
print(dataset.shape)
print(dataset.columns)

feature_cols = ['Pregnancies','Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = dataset[feature_cols]
y = dataset.Outcome

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
logreg= LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('------------')
print(y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True , cmap ='Blues_r', fmt= 'g' )
#ax.set_lable_position('top')
plt.tight_layout()
plt.title('Confussion matrix', y = 1.1)
plt.ylabel('Actual')
plt.xlabel('Predicciones')
plt.show()

#Interpretacion de MC 
#Arriba derecha cuales interpretamos correctamente. Caso diabes positivos correcto
#Abajo izq cuando es no diabetes correcto
#Arriba a la derecha no diabtes incorrectamente
#abajo derecha  diabetes incorrecta

print('Exactitud', metrics.accuracy_score(y_test,y_pred))