from flask import Flask, render_template
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    user = "Enrico"
    return render_template('index.html', name=user)