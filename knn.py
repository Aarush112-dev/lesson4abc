import pandas as pd
import numpy as np

data = pd.read_csv("Lesson 4 - KNN\data.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["species"] = le.fit_transform(data["species"])
data.info()


x = data[["sepal_length","sepal_width","petal_length","petal_width"]]
y = data["species"]

#from sklearn.preprocessing import StandardScalar
#ss = StandardScaler()
#x = ss.fit_transform(x)

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
x = mm.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state= 5)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
from matplotlib import pyplot as plt

matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test,y_pred))

