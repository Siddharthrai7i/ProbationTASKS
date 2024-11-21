from flask import Flask, render_template, request
import pickle
import numpy as np
# import scikit-learn as sklearn
model=pickle.load(open('classifier.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])


def predictthe():
    pregnancies=int(request.form.get('Pregnancies'))
    glucose=int(request.form.get('Glucose'))
    bloodpressure=int(request.form.get('BloodPressure'))
    skinthickness=int(request.form.get('SkinThickness'))
    insulin=float(request.form.get('Insulin'))
    bmi=float(request.form.get('BMI'))
    age=int(request.form.get('Age'))

    result =model.predict(np.array([pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,age]).reshape(1,7))
    return str(result)
if __name__ == '__main__':
    app.run(debug=True)