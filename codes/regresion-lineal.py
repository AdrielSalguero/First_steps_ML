import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('C:/Users/Usuario/Documents/Codes/ML-Fundamentos/data/salarios.csv')
print(dataset.head(10))
print(dataset.shape)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,:1].values
 
#DIVIDIMOS LA INFORMACION
#test size 70% del modelo random state 0 significa significa que no vamos a modificar los datos
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 0)

print(x_train, '------\n',y_train)

regressor = LinearRegression()
#Creacion del modelo con el .fit
regressor.fit(x_train,y_train)

viz_train = plt

viz_train.scatter(x_train, y_train, color = 'Blue')
#regressor predict es lo que trae la info que queremos ver
viz_train.plot(x_train, regressor.predict(x_train), color = 'Red')
viz_train.title('Salario / Experiencia')
viz_train.xlabel('Experiencia')
viz_train.xlabel('Salario')
viz_train.show()


viz_train = plt

viz_train.scatter(x_test, y_test, color = 'red')
#regressor predict es lo que trae la info que queremos ver
viz_train.plot(x_train, regressor.predict(x_train), color = 'black')
viz_train.title('Salario / Experiencia')
viz_train.xlabel('Experiencia')
viz_train.xlabel('Salario')
viz_train.show()


print(regressor.score(x_test, y_test))