## this is a logistic regression model to predict employee attrition based on the IBM HR dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.xls")
df.head()

df.shape

df.info()

df["Attrition"].value_counts()

round(df["Attrition"].value_counts(normalize=True) * 100,2)

df.duplicated().sum()

df.nunique()[df.nunique() == 1]  #constant columns



sns.boxplot(data=df, x="Attrition", y="Age")

nums_cols = ["Age",
"DailyRate",
"DistanceFromHome",
"HourlyRate",
"MonthlyIncome",
"MonthlyRate",
"NumCompaniesWorked",
"TotalWorkingYears",
"TrainingTimesLastYear",
"YearsAtCompany",
"YearsInCurrentRole",
"YearsSinceLastPromotion",
"YearsWithCurrManager",
"PercentSalaryHike"]

ordian_num_plt = ["Education",
"EnvironmentSatisfaction",
"JobInvolvement",
"JobLevel",
"JobSatisfaction",
"PerformanceRating",
"RelationshipSatisfaction",
"StockOptionLevel",
"WorkLifeBalance"]



round(pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100,2)

round(pd.crosstab(
    df["BusinessTravel"],
    df["Attrition"],
    normalize="index"
) * 100,2)

round(pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100,2)

round(pd.crosstab(
    df["Gender"],
    df["Attrition"],
    normalize="index"
) * 100,2)

pd.crosstab(
    df["Gender"],
    df["Attrition"])

round(pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
)*100,2)

round(pd.crosstab(df["MaritalStatus"],
                  df["Attrition"],
                  normalize="index")*100,2)

round(pd.crosstab(df["Over18"],
                  df["Attrition"],
                  normalize="index")*100,2)

for col in nums_cols:
  plt.figure(figsize=(10,5))
  sns.boxplot(data=df, x="Attrition", y=col)
  plt.show()

df.groupby('Attrition')['MonthlyIncome'].describe()

df.columns

drop_cols = ["Attrition","Over18","StandardHours","EmployeeNumber","EmployeeCount"]

X = df.drop(columns=drop_cols)
y = df["Attrition"].map({
    "Yes":1,
    "No":0
})





categorical_cols = X.select_dtypes(include="object").columns.tolist()

numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

print(categorical_cols)





preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

#---------------------------------train test split---------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(type(X_train))
print(X_train.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(round(accuracy*100,2))

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

