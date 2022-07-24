from flask import Flask,request
from dorami import chatbot

app = Flask(__name__)
 
@app.route('/')
def hello_name():
   return 'Hello World'
 
if __name__ == '__main__':
   app.run()