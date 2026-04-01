from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    exp = float(request.form['experience'])
    age = float(request.form['age'])
    
    education = request.form['education']
    work_type = request.form['work_type']

    # Encode inputs
    education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
    work_map = {"Onsite": 0, "Remote": 1}

    edu_val = education_map[education]
    work_val = work_map[work_type]

    prediction = model.predict([[exp, age, edu_val, work_val]])

    formatted_salary = f"Rs {int(prediction[0]):,}"
    return render_template('index.html', result=formatted_salary)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)