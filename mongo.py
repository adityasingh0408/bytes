from flask import Flask, render_template, request
from flask_pymongo import PyMongo

app = Flask(__name__)

# Replace with your MongoDB URI
app.config["MONGO_URI"] = "mongodb://localhost:27017"
mongo = PyMongo(app)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

    
