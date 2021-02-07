# Pytorch-Model-Inference

Objective: Deploy a “ML Prediction as a Service” System

#### How to run this project
```bash
FLASK_ENV=development FLASK_APP=app.py flask run
```

#### pre-request
```bash
torchvision 0.8.2
flask 1.1.2
requests 2.25.1
sqlite3
```

### How to query API
resp = requests.post("http://localhost:5000/predict",files={"file": open('{path_to_file}/dog.jpg','rb')})
resp.json()
