import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
dib_data=load_diabetes()
df=pd.DataFrame(data=dib_data.data,columns=dib_data.feature_names)
print(df.head())
X=dib_data.data[:,2].reshape(-1,1)
y=dib_data.target.reshape(-1,1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
model_with_bias=LinearRegression(fit_intercept=True)
model_with_bias.fit(X_train,y_train)
y_pred_with_bias=model_with_bias.predict(X_test)
mse_with_bias=mean_squared_error(y_test,y_pred_with_bias)
plt.scatter(X_test,y_test,label='Actual')
plt.plot(X_test,y_pred_with_bias,label='With Bias', color='red')
plt.xlabel('BMI')
plt.ylabel('Diabaetes Progression')
plt.title('Regression Model With Bias')
plt.legend()
plt.show()
print('Mean Squared Error (WITH BIAS):',mse_with_bias)
