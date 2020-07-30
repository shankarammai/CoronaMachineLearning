# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:51:16 2020

@author: Shankar
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#Importing the data file
data=pd.read_csv("covid_19_data.csv")
datacodes=pd.read_csv("codes.csv")
hdidata=pd.read_csv("HDI.csv")
#print Basic info about the data
print(data.info())
print("\n \n ###############Empty Data Check##########")
print(data.isnull().sum())


data.rename(columns={"Country/Region":"Country"},inplace=True)
data["DayofYear"] = pd.to_datetime(data["ObservationDate"]).dt.dayofyear
data.drop(['ObservationDate'], axis=1, inplace=True)


LE = LabelEncoder()
data['countrycode'] = LE.fit_transform(data['Country'])
countrycodes=data['countrycode']
dataGrouppedByCountry=data.groupby(['Country','DayofYear']).sum().reset_index()

dataGrouppedByCountry['Country'] = np.where(dataGrouppedByCountry['Country']=="Mainland China", 'China',dataGrouppedByCountry['Country'])
dataGrouppedByCountry.drop('SNo', axis=1, inplace=True)
#dataGrouppedByCountry["Date"] = pd.to_datetime(dataGrouppedByCountry["DayofYear"])#.dt.strftime("%m%d").astype(int)
#dataGrouppedByCountry.set_index('Date', inplace=True)



newdf = pd.merge(dataGrouppedByCountry, datacodes, how='left', left_on='Country', right_on='Country ')

newdf.drop(['index','Country '], axis=1, inplace=True)


hdidf=pd.merge(dataGrouppedByCountry, hdidata, how='left', left_on='Country', right_on='Country')
hdidf.drop(['HDI Rank (2018)','countrycode'], axis=1, inplace=True)
hdidf.rename(columns={"2018":"HDI"},inplace=True)
hdidf.dropna(inplace=True)
hdidf['HDI']=hdidf['HDI'].astype(float)

hdihigh= hdidf[hdidf.HDI>0.7]
hdilow=hdidf[hdidf.HDI<0.5]

hdihigh=hdihigh.groupby(['DayofYear']).sum().reset_index()
hdilow=hdilow.groupby(['DayofYear']).sum().reset_index()




plt.figure(figsize = (14, 10))
sns.heatmap(hdidf.corr(), annot = True)
plt.show()

plt.figure(figsize = (9,9))
plt.plot(range(hdihigh.shape[0]),(hdihigh['Deaths']),label="HDI High Countries")
plt.plot(range(hdilow.shape[0]),(hdilow['Deaths']),label="HDI LOw Countries")
plt.xticks(range(0,hdihigh.shape[0],5),hdihigh['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of Year',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title("Death Compare of HDI HIGH and LOW countires")
plt.legend(loc="upper left")
plt.show()


plt.figure(figsize = (9,9))
plt.plot(range(hdihigh.shape[0]),(hdihigh['Confirmed']),label="HDI High Countries")
plt.plot(range(hdilow.shape[0]),(hdilow['Confirmed']),label="HDI LOw Countries")
plt.xticks(range(0,hdihigh.shape[0],5),hdihigh['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of Year',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title("Confirmed cases Compare of HDI HIGH and LOW countires")
plt.legend(loc="upper left")
plt.show()


plt.figure(figsize = (9,9))
plt.plot(range(hdihigh.shape[0]),(hdihigh['Recovered']),label="HDI High Countries")
plt.plot(range(hdilow.shape[0]),(hdilow['Recovered']),label="HDI LOw Countries")
plt.xticks(range(0,hdihigh.shape[0],5),hdihigh['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of Year',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title("Recovered cases Compare of HDI HIGH and LOW countires")
plt.legend(loc="upper left")
plt.show()


plt.figure(figsize = (9,9))
#plt.plot(range(hdihigh.shape[0]),(hdihigh['Recovered']),label="HDI High Countries")
plt.plot(range(hdilow.shape[0]),(hdilow['Recovered']),label="HDI LOw Countries")
plt.xticks(range(0,hdihigh.shape[0],5),hdihigh['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of Year',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title("Recovered cases Compare of HDI HIGH and LOW countires")
plt.legend(loc="upper left")
plt.show()




dataGroupedByDate=data.groupby(['DayofYear']).sum().reset_index()
dataGroupedByDate.drop('SNo', axis=1, inplace=True)

Usercountry=input("Enter Country Name to visualise -> ")
choosenCountry = dataGrouppedByCountry[dataGrouppedByCountry['Country'].isin([Usercountry])]

plt.figure(figsize = (9,9))
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Deaths']),label="Deaths")
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Recovered']),label="Recovered")
plt.xticks(range(0,choosenCountry.shape[0],5),choosenCountry['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of Year',fontsize=18)
plt.ylabel('People',fontsize=18)
plt.title(Usercountry +" Deaths and Recovered cases")
plt.legend(loc="upper left")
plt.show()

plt.figure(figsize = (9,9))
plt.plot(range(choosenCountry.shape[0]),(choosenCountry['Confirmed']))
plt.xticks(range(0,choosenCountry.shape[0],5),choosenCountry['DayofYear'].loc[::5],rotation=270)
plt.xlabel('Day of year',fontsize=18)
plt.ylabel('Deaths',fontsize=18)
plt.title(Usercountry +" Deaths")
plt.show()






#choosenCountry.drop(['Country','countrycode'], axis=1, inplace=True)
#x=choosenCountry.loc[:, choosenCountry.columns != 'Confirmed']
#y=choosenCountry['Confirmed']
#X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.9,shuffle=False) 
#
#from sklearn.ensemble import RandomForestRegressor
#import time
#RandomForestRF = RandomForestRegressor(random_state=0, n_estimators=150)
#startTime=time.time()
#RandomForestRF.fit(X_train, y_train)
#predections=RandomForestRF.predict(X_test)
#predections=pd.DataFrame(predections)
#endTime=time.time()
#print("Time Taken to Train -> ",endTime-startTime)
#print("Predection Accuracy Score -> ",RandomForestRF.score(X_test, y_test)*100)






#Compare = pd.DataFrame({'predections' : [],'actual':[]})
#Compare['predections']= pd.Series(predections)
#Compare['actual']=pd.Series(y_test.reset_index())




#
#
#from statsmodels.tsa.api import VAR
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tools.eval_measures import rmse, aic
#
#
#
#from statsmodels.tsa.vector_ar.vecm import coint_johansen
#testa=coint_johansen(dataGrouppedByCountry,-1,1).eig
#
##creating the train and validation set
#train = choosenCountry[:int(0.8*(len(choosenCountry)))]
#valid = choosenCountry[int(0.8*(len(choosenCountry))):]
#
##fit the model
#from statsmodels.tsa.vector_ar.var_model import VAR
#
#model = VAR(endog=train)
#model_fit = model.fit()
#
## make prediction on validation
#prediction = model_fit.forecast(model_fit.y, steps=len(valid))
#
#cols = choosenCountry.columns
#pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
#for j in range(0,3):
#    for i in range(0, len(prediction)):
#       pred.iloc[i][j] = prediction[i][j]
#
##check rmse
#for i in cols:
#    print('rmse value for', i, 'is : ', np.sqrt(mean_squared_error(pred[i], valid[i])))
#
#
#
#
