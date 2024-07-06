#USING RANDOM FOREST REGRESSOR

#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error



pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

df = pd.read_csv('Housing.csv')

print(df.head())
print(df.shape)
print(df.info())
#print(df.isnull().sum())


#separating categorical and numerical data
cat_col = []
num_col = []

for i in df.columns:
    if df[i].dtype == object:
        cat_col.append(i)
    else:
        num_col.append(i)

print(cat_col)
print(num_col)


#handle categorical values-> numerical by encoding
le = LabelEncoder()
for col in cat_col:
    df[col] = le.fit_transform(df[col])

print(df.head())

#train test split
y = df['price']
X = df.drop(columns='price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=766)

print(X_train.shape, X_test.shape) #(436, 12) (109, 12)


#model
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)

#metrics
rfr_error = mean_absolute_error(y_test, rfr_prediction)
rfr_error_percentage = mean_absolute_percentage_error(y_test, rfr_prediction)
print(rfr_error, rfr_error_percentage*100) 

#scatter plot

#plot actual vs predicted
actual = y_test
predicted = rfr_prediction

plt.figure(figsize=(15, 10))

# Plot the actual values as a scatter plot
plt.scatter(range(len(actual)), actual, color='blue', label='Actual')

# Plot the predicted values as a line
plt.scatter(range(len(actual)), predicted, color='red', label='Predicted')

# A line between the actual point and predicted point
for i in range(len(actual)):
    plt.plot([i, i], [actual.iloc[i], predicted[i]], color='green', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (House price prediction)')
plt.legend()
plt.show()