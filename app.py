from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.externals import joblib
import pickle

# load the model from disk
model1 = 'Model1.pkl'
clf1 = pickle.load(open(model1, 'rb'))

model2 = 'Model2.pkl'
clf2 = pickle.load(open(model2, 'rb'))

group_dict=pickle.load(open('Group_label_mapping.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        Description = pd.Series(request.form['Description'])
        my_prediction = clf1.predict(Description)
        if(my_prediction==0):
            output = 'GRP_0'
        elif(my_prediction==1):
            op= clf2.predict(Description)
            output = group_dict[op[0]]
    return render_template('result.html', output = output)



if __name__ == '__main__':
	app.run(debug=True)
