import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from pydataset import data
import pandas as pd
import numpy as np
import acquire

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def prep_iris():
    iris = acquire.get_iris_data()
    iris.drop(columns=['species_id', 'measurement_id', 'species_id.1'], inplace = True)
    iris.rename(columns={"species_name":"species" }, inplace = True)
    iris_dummies = pd.get_dummies(iris['species'])
    iris = pd.concat([iris, iris_dummies], axis=1)
    return iris

def prep_titanic():
    titanic = acquire.get_titanic_data()
    titanic = titanic[~titanic.embark_town.isnull()]
    titanic.drop(columns=['deck'])
    titanic_dummies = pd.get_dummies(titanic['embarked'])
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    scaler = MinMaxScaler()
    scaler = MinMaxScaler()
    titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])
    imputer = SimpleImputer(strategy = 'mean')
    imputer = imputer.fit(titanic[['age']])
    titanic[['age']] = imputer.transform(titanic[['age']])
    return titanic