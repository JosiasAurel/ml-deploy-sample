
from tensorflow.keras import models
from flask import Flask, request

app = Flask(__name__)

# load saved model from filesystem
model = models.load_model(filepath="./")

@app.route('/predict', methods=["POST"])
def predict():
	value = int(request.data)
	prediction = model.predict([value])
	print(prediction)
	return str(prediction[0][0])

if __name__ == "__main__":
	app.run(debug=True)

# print(model.predict([8]))
