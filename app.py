from flask import Flask, request
import json
import urllib.request
from fastai.vision import *
from io import BytesIO


UPLOAD_FOLDER = '/home/raza/Desktop'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/analyze", methods = ['GET', 'POST'])
def home():
	if request.method == "POST":
		# read the image from the incoming post request
		image = request.files["file"].read()
		# setup for pytorch to run the fastai model
		defaults.device = torch.device('cpu')
		path = Path('.')
		# load the model into memory
		learner = load_learner(path, 'export.pkl')
		# Open the image in the form of bytes using BytesIO library
		img = open_image(BytesIO(image))
		# Predict the image and pick the prediction that has the highest probabality 
		prediction = learner.predict(img)[0]
		# print(str(prediction))
		return {'result': str(prediction)}
	return "Render Application"


if __name__ == "__main__":
	app.run(threaded = True)
