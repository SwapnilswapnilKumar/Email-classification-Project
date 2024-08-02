from flask import Flask, url_for,render_template,request
import joblib
bow_obj = joblib.load('./models/bag_of_words.lb')
bnb = joblib.load('./models/bernoulliNB.lb')

app=Flask(__name__)


@app.route('/')
def home():
    # return "welcome to flask"
    return render_template('home.html')

@app.route('/project')
def project():
    return render_template('index.html')

@app.route('/predictions',methods=["POST","GET"])
def prediction():
    res=""
    if request.method=="POST":
        message = str(request.form['message'])
        email_message = [message]
        email_converted = bow_obj.transform(email_message).toarray()
        prediction = bnb.predict(email_converted)
        
        if prediction==1:
            res="SPAM"
        else:
            res="HAM"

    return render_template('final.html',output=res)



if __name__=="__main__":
    app.run(debug=True)