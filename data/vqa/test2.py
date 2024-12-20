import json

f = open('valid.json')
data = json.load(f)
print(len(data))