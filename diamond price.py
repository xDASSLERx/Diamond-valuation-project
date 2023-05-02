import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluation(X_train, y_train, X_test, y_test, model, size):
    predict = X_train @ model.coef_ + model.intercept_
    error = predict - y_train
    [x for p, x in zip(predict, X_train) if p<0]
    err_sorted = sorted(zip(error, X_train, predict), key=lambda q:q[1][0])
    err, _, _ = zip(*err_sorted)
    plt.plot(err, color="b")
    plt.show()

    print(size, "Train dataset", (np.abs(predict-y_train)/(predict+y_train)).mean())
    predict_valid = X_test @ model.coef_ + model.intercept_
    print(size, "Test dataset",(np.abs(predict_valid-y_test)/(predict_valid+y_test)).mean())

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(size, 'Mean squared error:', mse)
    print(size, 'R-squared:', r2)

df = pd.read_csv('diamonds.csv')

print(df.describe())
df = df[(df["x"]>0) & (df["y"]>0) & (df["z"]>0)]
print(df.corr())
df.drop(['x', 'y', "z"], axis= 1 , inplace = True ) 

sb.histplot(data=df["price"], stat="count")
plt.show()
sb.histplot(data=df["carat"], stat="count")
plt.show()

df_big = df[df["carat"]>2.3]
df = df[df["carat"]>0.2]
df_small = df[df["carat"]<=2.3]
print(df_small.describe())
print(df_big.describe())

sb.histplot(data=df_small["price"], stat="count")
plt.show()
sb.histplot(data=df_small["carat"], stat="count")
plt.show()
sb.histplot(data=df_big["price"], stat="count")
plt.show()
sb.histplot(data=df_big["carat"], stat="count")
plt.show()

selection_small = df_small[df_small["carat"]>=0.3]
selection_small = selection_small[df["carat"]<=0.35]
sb.catplot(data=selection_small, x="clarity", y="price", kind="point")
sb.catplot(data=selection_small, x="cut", y="price", kind="point")
sb.catplot(data=selection_small, x="color", y="price", kind="point")
plt.show()

selection_big = df_big[df_big["carat"]>=2.3]
sb.catplot(data=selection_big, x="clarity", y="price", kind="point")
sb.catplot(data=selection_big, x="cut", y="price", kind="point")
sb.catplot(data=selection_big, x="color", y="price", kind="point")
plt.show()

cut_small = {'Very Good':640, 'Premium':730, 'Fair':840, 'Good':580, 'Ideal':760}
clarity_small = {'VS1':690, 'VVS2':790, 'SI1':580, 'SI2':495, 'I1':600, 'VVS1':860, 'IF':950, 'VS2':700}
color_small = {'G':740, 'J':450, 'F':760, 'I':550, 'H':650, 'D':760, 'E':780}
df_small = df_small.replace(clarity_small)
df_small = df_small.replace(cut_small)
df_small = df_small.replace(color_small)

cut_big = {'Very Good':16300, 'Premium':16000, 'Fair':13700, 'Good':16300, 'Ideal':16000}
clarity_big = {'VS1':16900, 'VVS2':15400, 'SI1':16900, 'SI2':16000, 'I1':8500, 'VVS1':16000, 'IF':15400, 'VS2':17500}
color_big = {'G':13500, 'J':16400, 'F':17600, 'I':16200, 'H':15400, 'D':16300, 'E':14800}
df_big = df_big.replace(clarity_big)
df_big = df_big.replace(cut_big)
df_big = df_big.replace(color_big)

print(df_small.corr()["price"])
print(df_big.corr()["price"])

df_small['per-carat'] = np.log(df_small['price']/df_small['carat']**2) 
X_small = df_small[['carat', 'color', 'clarity', 'table']].values
Y_small = df_small['per-carat'].values
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_small, Y_small, test_size=0.25)
model_small = LinearRegression()
model_small.fit(X_train_small, y_train_small)
print("Small", model_small.intercept_, model_small.coef_)

df_big['per-carat'] = np.log(df_big['price']/df_big['carat']**2) 
X_big = df_big[['carat', 'color', 'clarity', 'table', "depth", ]].values
Y_big = df_big['per-carat'].values
X_train_big, X_test_big, y_train_big, y_test_big = train_test_split(X_big, Y_big, test_size=0.25)
model_big = LinearRegression()
model_big.fit(X_train_big, y_train_big)
print("Big", model_big.intercept_, model_big.coef_)

evaluation(X_train_small, y_train_small, X_test_small, y_test_small, model_small, "Small diamonds")
evaluation(X_train_big, y_train_big, X_test_big, y_test_big, model_big, "Big diamonds")




