import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st

df=pd.read_excel('cars.xls')
X=df.drop('Price',axis=1)
y=df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor=ColumnTransformer(
	transformers=[
		('num',StandardScaler(),['Mileage','Cylinder','Liter','Doors']),
		('cat',OneHotEncoder(),['Make','Model','Type'])	
	]
	)
model=LinearRegression()
pipeline=Pipeline(steps=[('preprocessor',preprocessor),('model',model)])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
rmse=mean_squared_error(y_test,pred,squared=False)
r2=r2_score(y_test,pred)

def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
	input_data=pd.DataFrame({
		'Make':[make],
		'Model':[model],
		'Trim':[trim],
		'Mileage':[mileage],
		'Type':[car_type],
		'Car_type':[car_type],
		'Cylinder':[cylinder],
		'Liter':[liter],
		'Doors':[doors],
		'Cruise':[cruise],
		'Sound':[sound],
		'Leather':[leather]
		})
	prediction=pipeline.predict(input_data)[0]
	return prediction
st.title("Car Price Prediction :red_car: @drmurataltun")
st.write("Enter Car Details to predict the price of the car")
make=st.selectbox("Make",df['Make'].unique())
model=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim=st.selectbox("Trim",df[(df['Make']==make) & (df['Model']==model)]['Trim'].unique())
mileage=st.number_input("Mileage",200,60000)
car_type=st.selectbox("Type",df['Type'].unique())
cylinder=st.selectbox("Cylinder",df['Cylinder'].unique())
liter=st.number_input("Liter",1,6)
doors=st.selectbox("Doors",df['Doors'].unique())
cruise=st.radio("Cruise",[True,False])
sound=st.radio("Sound",[True,False])
leather=st.radio("Leather",[True,False])
if st.button("Predict"):
	pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)
	st.write("Predicted Price :red_car:  $",round(pred[0],2))