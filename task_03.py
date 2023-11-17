import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import graphviz

data = pd.read_csv('diabetes.csv', delimiter=',', decimal='.')
data_values = data.values
data_values = np.array(list(data_values))

OUTCOME_COLUMN = 8

train_data = data_values[:520, ]
not_ill_person = sum([x == 0 for x in train_data[:, OUTCOME_COLUMN]])
print("Количество здоровых пациентов (класс 0): ", not_ill_person)

#разделение выборки 80% - тренировочный надор, 20% - тестовый
train_data_length = len(train_data)
len_80_per = int(train_data_length * 0.8)

X_train = train_data[:len_80_per, :OUTCOME_COLUMN]
y_train = train_data[:len_80_per, OUTCOME_COLUMN]
X_test = train_data[len_80_per: , :OUTCOME_COLUMN]
y_test = train_data[len_80_per: , OUTCOME_COLUMN]

classifer = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=25, min_samples_leaf=15, random_state=2020)
classifer.fit(X_train, y_train)

score = classifer.score(X_test, y_test)
print("Доля правильных ответов: ", score)

f1 = metrics.f1_score(y_test, classifer.predict(X_test), average='macro')
print("f1: ", f1)

predict_idx = [719, 739, 748, 734]

for i in range(len(predict_idx)):
  X = data_values[predict_idx[i], :OUTCOME_COLUMN]
  print(f"Предсказание класса для пациента {predict_idx[i]}: {int(classifer.predict([X])[0])}")

#построение дерева
columns = list(data.columns)[:OUTCOME_COLUMN]
export_graphviz(classifer, 
                out_file = 'tree1.dot', 
                feature_names = columns, 
                class_names  =['0', '1'], 
                rounded = True,
                proportion = False,
                precision = 4, 
                filled = True,
                label = 'all')

with open('tree1.dot') as f:
  dot_graph = f.read()

graphviz.Source(dot_graph)
