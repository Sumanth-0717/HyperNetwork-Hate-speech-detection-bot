from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
import numpy as np

app = Flask(__name__, template_folder='templates')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tweets.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

dt = joblib.load(r'models/trained_model.sav')
cv = joblib.load(r'models/count_vectorizer.sav')

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(280), nullable=False)

@app.route('/')
def index():
    tweets = Tweet.query.all()
    return render_template('index.html', tweets=tweets)

@app.route('/post')
def post():
    tweets = Tweet.query.all()
    return render_template('post.html', tweets=tweets)

@app.route('/post_tweet', methods=['POST'])
def post_tweet():
    try:
        text = request.form['tweet']
        if not text:
            return render_template("post.html", prediction="No text provided")
        
        transformed_text = cv.transform([text]).toarray()

        prediction = dt.predict(transformed_text)

        prediction_str = prediction[0]
        
        if prediction_str == 'no hate or offensive language': 
            new_tweet = Tweet(content=text)
            db.session.add(new_tweet)
            db.session.commit()
            return render_template("post.html", prediction="Post saved", tweets=Tweet.query.all())
        else:
            return render_template("post.html", prediction="Hate speech detected, Post not saved", tweets=Tweet.query.all())
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template("post.html", prediction="Prediction failed", tweets=Tweet.query.all())

@app.route('/profile')
def profile():
    tweets = Tweet.query.all()
    return render_template("profile.html", tweets=tweets)

@app.route('/delete_tweet/<int:tweet_id>', methods=['POST'])
def delete_tweet(tweet_id):
    try:
        tweet = Tweet.query.get_or_404(tweet_id)
        db.session.delete(tweet)
        db.session.commit()
        return redirect(url_for('profile'))
    except Exception as e:
        print(f"Error deleting tweet: {str(e)}")
        return render_template("index.html", prediction="Error deleting tweet", tweets=Tweet.query.all())


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  
    app.run(debug=True)
