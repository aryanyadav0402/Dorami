#from dorami import chatbot
from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
 
# def __init__(self):
#     	self.chatbot = chatbot()


@app.route("/", methods=["POST"])
def hello_name():
	input_string = request.get_json()['input_string']
	#res = chatbot(input_string) # res should be a dict
	res={
		"data": "These are the laptop",
		"title":["HP Pavilion Ryzen 5 Hexa Core 5600H - (8 GB/512 GB SSD/Windows 10/4 GB Graphics/NVIDIA GeForce GTX 1650/144 Hz) 15-ec2004AX Gaming Laptop  (15.6 inch, Shadow Black, 1.98 kg)","ASUS TUF Gaming A17 Ryzen 7 Octa Core 4800H - (16 GB/512 GB SSD/Windows 10 Home/4 GB Graphics/NVIDIA GeForce RTX 3050/144 Hz) FA706IC-HX003T Gaming Laptop  (17.3 inch, Graphite Black, 2.60 kg)"] ,
		"price" : ["100","200"] ,
		"img_url": ["https://rukminim1.flixcart.com/image/612/612/kbqu4cw0/computer/q/x/r/hp-original-imaftyzachgrav8f.jpeg?q=70","https://rukminim1.flixcart.com/image/612/612/l3rmzrk0/computer/z/2/c/-original-imagetjyhhtrtkdg.jpeg?q=70"] ,
		"title_url":["https://www.flipkart.com/hp-pavilion-ryzen-5-hexa-core-5600h-8-gb-512-gb-ssd-windows-10-4-graphics-nvidia-geforce-gtx-1650-144-hz-15-ec2004ax-gaming-laptop/p/itm98c94bbf9bc20?pid=COMG5GZXPWMGTNWS&lid=LSTCOMG5GZXPWMGTNWSQE9WVW&marketplace=FLIPKART&q=HP+Pavilion+Ryzen+5+Hexa+Core+5600H+-+%288+GB%2F512+GB+SSD%2F...%3B+ASUS+TUF+Gaming+A17+Ryzen+7+Octa+Core+4800H+-+%2816+GB%2F51...%3B+acer+Aspire+7+Core+i5+10th+Gen+-+%288+GB%2F512+GB+SSD%2FWindo...%3B+ASUS+TUF+Gaming+F15+Core+i5+10th+Gen+-+%288+GB%2F512+GB+SSD...&store=search.flipkart.com&srno=s_1_2&otracker=search&otracker1=search&fm=Search&iid=7788951f-4646-47a8-a746-6239f371432a.COMG5GZXPWMGTNWS.SEARCH&ppt=sp&ppn=sp&ssid=xcyp1j9xj40000001656243820753&qH=707549e6ce269adf",
						"https://www.flipkart.com/asus-tuf-gaming-a17-ryzen-7-octa-core-4800h-16-gb-512-gb-ssd-windows-10-home-4-graphics-nvidia-geforce-rtx-3050-144-hz-fa706ic-hx003t-laptop/p/itmccc79bbd17e07?pid=COMG8N5PFPCX9Z3J&lid=LSTCOMG8N5PFPCX9Z3JGKCEFC&marketplace=FLIPKART&cmpid=content_computer_15083003945_u_8965229628_gmc_pla&tgi=sem,1,G,11214002,u,,,556262839325,,,,c,,,,,,,&gclid=CjwKCAjwh-CVBhB8EiwAjFEPGcxYUKUL4RQPdprBhLDbEgKm7R14pHuY0o73wTzRXQ-amw29uNg3XBoC5mIQAvD_BwE"]		
	}
	
	return res

if __name__ == '__main__':
	app.run(debug=True)