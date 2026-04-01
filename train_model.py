import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "experience": [1,2,3,4,5,6,7,8],
    "age": [22,25,28,30,35,40,45,50],
    "education": ["Bachelors","Bachelors","Masters","Masters","PhD","PhD","PhD","PhD"],
    "work_type": ["Onsite","Remote","Onsite","Remote","Remote","Onsite","Remote","Onsite"],
    "salary": [30000,36000,40000,47000,55000,60000,68000,72000]
}

df = pd.DataFrame(data)

# Encoding
df['education'] = df['education'].map({
    "Bachelors": 0,
    "Masters": 1,
    "PhD": 2
})

df['work_type'] = df['work_type'].map({
    "Onsite": 0,
    "Remote": 1
})

X = df[['experience', 'age', 'education', 'work_type']]
y = df['salary']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model retrained!")