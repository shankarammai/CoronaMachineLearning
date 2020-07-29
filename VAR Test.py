# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:51:16 2020

@author: Shankar
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Importing the data file
data=pd.read_csv("covid_19_data.csv")

#print Basic info about the data
print(data.info())
print("\n \n ###############Empty Data Check##########")
print(data.isnull().sum())


data.rename(columns={"Country/Region":"Country"},inplace=True)
data["Date"] = pd.to_datetime(data["ObservationDate"]).dt.strftime("%d%m%Y")

LE = LabelEncoder()
data['countrycode'] = LE.fit_transform(data['Country'])
dataGrouppedByCountry=data.groupby(['Country','ObservationDate']).sum().reset_index()

dataGrouppedByCountry['Country'] = np.where(dataGrouppedByCountry['Country']=="Mainland China", 'China',dataGrouppedByCountry['Country'])
dataGrouppedByCountry.drop('SNo', axis=1, inplace=True)


dataGroupedByDate=data.groupby(['ObservationDate']).sum().reset_index()
dataGroupedByDate.drop('SNo', axis=1, inplace=True)

Usercountry=input("Enter Country Name to visualise -> ")
choosenCountry = dataGrouppedByCountry[dataGrouppedByCountry['Country'].isin([Usercountry])]

plt.figure(figsize = (9,9))
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Deaths']),label="Deaths")
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Recovered']),label="Recovered")
plt.xticks(range(0,choosenCountry.shape[0],5),choosenCountry['ObservationDate'].loc[::5],rotation=270)
plt.xlabel('ObservationDate',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title(Usercountry +" Deaths and Recovered cases")
plt.legend(loc="upper left")
plt.show()

plt.figure(figsize = (9,9))
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Confirmed']))
plt.xticks(range(0,choosenCountry.shape[0],5),choosenCountry['ObservationDate'].loc[::5],rotation=270)
plt.xlabel('ObservationDate',fontsize=18)
plt.ylabel('Deaths',fontsize=18)
plt.title(Usercountry +" Deaths")
plt.show()

#data.drop(['ObservationDate','Country'], axis=1, inplace=True)
#dataGrouppedByCountry.drop(['ObservationDate','Country'], axis=1, inplace=True)




    
    







