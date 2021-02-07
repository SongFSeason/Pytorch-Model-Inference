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
```python
resp = requests.post("http://172.31.23.99:5000/predict",files={"file": open('/home/ubuntu/Pytorch-Model-Inference/215_imgnet_imgs/ILSVRC2012_test_00000013.JPEG','rb')})
resp.json()
```
