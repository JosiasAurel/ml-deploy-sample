--- 
title: Create and serve a simple machine learning model
tags: [Machine Learning, Python, API]
date: 27/03/2021
---

In this tutorial, we are going to create and deploy a simple machine learning model. 
We are going to create the model using Python and the tensorflow library. We will finish up by serving the model in a flask application as an API _(Application Programming Interface)_.

Let's get to it.

First create a directory/folder where you are going to put all the code from this tutorial.

The model we are going to create is a pretty simple one. Given an input _x_, we want to get an output _2x+1_. Pretty simple! We want to create a model that will train on some sample data and come up with its own method to be able to find the right output. 
Our sample data is pretty simple one.

```python
sample_input = (1, 2, 3, 4, 5, 6)
sample_output = (3, 5, 7, 9, 11, 13)
```
If you try using the formula on the input, you should get a corresponding output.

Now open the project folder in your favorite editor and let's get to coding.

## Creating the model

Create a new file called _model.py_ and we are going ro write our model code in there.

First we are going to import tensorflow and keras.

```python 
import tensorflow as tf
from tensorflow import keras
```
Next is to create out model instance. Since it is simple one, a sequential model will be fine.

```python 
model = keras.Sequential([
  keras.layers.Dense(1, activation="relu", input_shape=[1])
])
```

Next step is to compile our model and give it an optmizer as well as a loss function. The loss function will help to look back at how the model progresses from errors while rhe optmizer will help optmize the model.

```python 
model.compile(optimizer="sgd", loss="mean_squared_error")
```

Next comes training our model. 
```python 
# train model
model.fit(sample_input, sample_output, epochs=500)
```

You should now end up with this
```python 
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
  keras.layers.Dense(1, activation="relu", input_shape=[1])
])

sample_input = (1, 2, 3, 4, 5, 6)
sample_output = (3, 5, 7, 9, 11, 13)

model.compile(optimizer="sgd", loss="mean_squared_error")

# train model
model.fit(sample_input, sample_output, epochs=500)

```
Time to test our model. Add this line at the end of the file. You can pass any value in it to see if the model works. Make sure its in the form of a list.

```python 
print(model.predict([8])) # my output [[17.074446]]
```
After training the model, we try to make it predict on a sample value. I choosed to test on 8 as my sample and i got ~17. If we substitute in our formula, 2(8)+1 equal 17. This confirms that our model actually works.

## Saving the model
After creating the model, it is good to save it so it can be deployed more easily.
Saving the model is as simple as using the _model.save()_ method.

Add this line at the end of your model.py file.
```python 
model.save(filepath="./")
```
The model.save() method takes a _filepath_ keyword argument which is the path to which you want to save the model. I am saving my model in the root of the project.
You should see a file named *saved_model.pb*. This is our model and can now be shared or deployed.

## Deploying the model

We are now going to create a simple flask API to serve our model. 

Create a new file named *app.py* and add the follwing code in there.

```python 
from tensorflow.keras import models
from flask import Flask, request

app = Flask(__name__)

# load saved model from filesystem
model = models.load_model(filepath="./")

@app.route('/predict', methods=["POST"])
def predict():
	value = int(request.data)
	prediction = model.predict([value])
	# print(prediction)
	return str(prediction[0][0])

if __name__ == "__main__":
	app.run(debug=True)
```
First, we are importing models utility from keras and then Flask.
We can load a saved model using the modeld utility using the load_model method. It takes a *filepath* argument specifying the path to which the saved model is.

Next is we create a route which accpet *POST* requests.
Inside the function, we get the data sent through the request. We will get bytes, so we cast it to an integer.
We then we used our loaded model to predict the output of the value sent. We then return our prediction. Notice we cast the response a string. This is because flask allows us to return only strings, tuples or dictionaries.

## Testing our API
To test your API, we are going to write a simple python program to do so.
Open a file called *test.py* and add the following code in it.

```python
import requests                         import sys

value = sys.argv[1]                                                                 res = requests.post("http://localhost:5000/predict", value)

print(res.content)
```

Testing it 
```shell
python test.py 9
```
I get *b'19.115118'*

You have reached the end of this tutorial.
I hope you enjoyed building and serving this simple model.

Feel free to play with the API with different values.


