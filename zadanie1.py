# import pandas
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np
import pylab


# read csv file into a DataFrame
water_train = pd.read_csv(r'C:\Users\Peter\Desktop\suns\data-Z1\water_train.csv')
print("Water train loaded.\n\n")
# display DataFrame
print(water_train.shape)
print("Shape printed.\n\n")

print(water_train.describe())
print("Describe printed.\n\n")

print(water_train.values)
print("Values printed.\n\n")

print(water_train)
print("Everything printed.\n\n")

# read csv file into a DataFrame
water_test = pd.read_csv(r'C:\Users\Peter\Desktop\suns\data-Z1\water_test.csv')
print("Water test loaded.\n\n")

# display DataFrame
print(water_test.shape)
print("Shape printed.\n\n")

print(water_test.describe())
print("Describe printed.\n\n")

print(water_test.values)
print("Values printed.\n\n")

print(water_test)
print("Everything printed.\n\n")

#normalize data
def normalize(file):
     imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
     imputer = imputer.fit(file[['ph']])
     file['ph']= imputer.transform(file[['ph']])
     imputer = imputer.fit(file[['Hardness']])
     file['Hardness']= imputer.transform(file[['Hardness']])
     imputer = imputer.fit(file[['Chloramines']])
     file['Chloramines']= imputer.transform(file[['Chloramines']])
     imputer = imputer.fit(file[['Sulfate']])
     file['Sulfate']= imputer.transform(file[['Sulfate']])
     imputer = imputer.fit(file[['Conductivity']])
     file['Conductivity']= imputer.transform(file[['Conductivity']])
     imputer = imputer.fit(file[['Organic_carbon']])
     file['Organic_carbon']= imputer.transform(file[['Organic_carbon']])
     imputer = imputer.fit(file[['Trihalomethanes']])
     file['Trihalomethanes']= imputer.transform(file[['Trihalomethanes']])
     imputer = imputer.fit(file[['Turbidity']])
     file['Turbidity']= imputer.transform(file[['Turbidity']])
     imputer = imputer.fit(file[['Potability']])
     file['Potability']= imputer.transform(file[['Potability']])
     print(file)
     normalized = file
     normalized = preprocessing.normalize(normalized)
     normalized = pd.DataFrame(normalized, columns = file.columns)
     return normalized
    
normalized_data = normalize(water_train)
print("Water train loaded.\n\n")

# display DataFrame
print(normalized_data.shape)
print("Shape printed.\n\n")


#show normalized training data
print(normalized_data)
print("Everything printed.\n\n")

pd.DataFrame(water_train).to_csv("foo.csv")

normalized_data.plot.density()
pylab.show()


