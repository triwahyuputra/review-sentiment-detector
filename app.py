from flask import Flask,render_template,url_for,request
import pickle

clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__, template_folder='home')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)