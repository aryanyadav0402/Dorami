from dorami import chatbot
from flask import Flask, request
app = Flask(__name__)
 
@app.route('/')
def hello_name():
	input_string = request.get_json()['input_string']
	res = chatbot(input_string) # res should be a dict
	# {
	# 	"data": "These are the laptop",
	# 	"img_url": "http..."
	# }
	return res

if __name__ == '__main__':
	app.run(debug=True)

curl --header "Content-Type: application/json" \
  --request GET \
  --data '{"input_string":"I want to buy dell laptop"}' \
  http://localhost:5000/