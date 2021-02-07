from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json
import io
from flask import Flask, jsonify, g, request
import datetime
import sqlite3




app = Flask(__name__)



DATABASE = 'prediction.db'


imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.squeezenet1_0(pretrained=True)
model.eval()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    with app.app_context():
        db=get_db()
        cur = db.cursor()
        try:
            tensor = transform_image(image_bytes=image_bytes)
            outputs = model.forward(tensor)
            _, y_hat = outputs.max(1)
            predicted_idx = str(y_hat.item())
            class_id, class_name = imagenet_class_index[predicted_idx]
            cur.execute("CREATE TABLE IF NOT EXISTS query_history (CLASS_ID TEXT, CLASS_NAME TEXT, SCORE_TIME TEXT);")
            cur.execute("INSERT INTO query_history VALUES (?, ?, ?);", (class_id, class_name, datetime.datetime.now()))

            db.commit()


            return jsonify({'class_id': class_id, 'class_name': class_name})
        except:
            db.rollback()
            db.close()
            return jsonify({'message':'application internal error'})



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        return get_prediction(image_bytes=img_bytes)


if __name__ == '__main__':
    app.run()