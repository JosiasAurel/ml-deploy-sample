
import requests
import sys

value = sys.argv[1]

res = requests.post("http://localhost:5000/predict", value)

print(res.content)
