from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import requests
import io
from flask import Flask, jsonify, abort, make_response, request
import os
from flask_sqlalchemy import SQLAlchemy
from flask import CORS

#### MTCNN ResNet のモデル読み込み
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#### 画像ファイルから画像の特徴ベクトルを取得(ndarray 512次元)
def feature_vector(url):
    img = Image.open(io.BytesIO(requests.get(url).content))
    img_cropped = mtcnn(img)
    feature_vector = resnet(img_cropped.unsqueeze(0))
    feature_vector_np = feature_vector.squeeze().to('cpu').detach().numpy().copy()
    return feature_vector_np

#### 2つのベクトル間のコサイン類似度を取得(cosine_similarity(a, b) = a・b / |a||b|)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def make_result(compare, urls):
    res = {}
    res['compare'] = compare
    for key in urls.keys():
        res[key] = {}
        for url in urls[key]:
            img1 = feature_vector(url)
            img2 = feature_vector(compare)
            res[key][url] = str(cosine_similarity(img1, img2))
    return res

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://feqrylwnedlfvi:c83988be335494405e41942747685f857107c79df24a844207c34d24b9f61f3a@ec2-52-70-205-234.compute-1.amazonaws.com:5432/d5qflmnn8ap0a5'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Flag(db.Model):
    __tablename__ = 'flag'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.Text)
    value = db.Column(db.Text)

    def __init__(self, key, value):
        self.key = key
        self.value = value

def get_flag_value_by_key(key):
    flag_list = db.session.query(Flag).filter(Flag.key == key).all()
    return flag_list[0].value

def register_flag(key, value):
    flag_list = db.session.query(Flag).filter(Flag.key == key).all()
    if len(flag_list) == 0:
        flag = Flag(key=key, value=value)
        db.create_all()
        db.session.add(flag)
        db.session.commit()
        return 'registerd'
    else:
        return 'error'

def update_flag(key, new_value):
    flag_list = db.session.query(Flag).filter(Flag.key == key).all()
    flag_list[0].value = new_value
    db.session.add_all(flag_list)
    db.session.commit()

def delete_flag(key):
    flag_list = db.session.query(Flag).filter(Flag.key == key).all()
    db.session.delete(flag_list[0])
    db.session.commit()

@app.route('/flag/register/<key>/<value>', methods=['GET'])
def register(key, value):
    res = register_flag(key, value)
    if res == 'registerd':
        return '{}:{}\nregisterd'.format(key, value)
    else:
        return 'error'

@app.route('/flag/read/<key>', methods=['GET'])
def read_flag(key):
    value = get_flag_value_by_key(key)
    return value

@app.route('/flag/update/<key>/<new_value>', methods=['GET'])
def updateFlag(key, new_value):
    update_flag(key, new_value)
    return 'updated'

@app.route('/flag/delete/<key>', methods=['GET'])
def deleteFlag(key):
    delete_flag(key)
    return 'deleted'


@app.route('/')
def hello():
    return 'hello facenet_pytorch'

@app.route('/post', methods=['POST'])
def post():
    data = request.get_json()
    compare = data['compare']
    urls = data['urls']
    res = make_result(compare, urls)
    return make_response(jsonify(res))

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.models import (
    MessageEvent, JoinEvent, TextMessage, TextSendMessage, LocationMessage, CarouselTemplate, CarouselColumn, TemplateSendMessage, URITemplateAction, ButtonsTemplate, PostbackTemplateAction, PostbackEvent, MessageAction
)
CHANNEL_ACCESS_TOKEN = 'k5tuhu/cYkLBG3M1UaS8I4WW7V40B1eNqKlgiE9bdM3OnqGPw3KvoTn+ZrIZplDHUVnUT/CfzM1Z/Peg8H7hYIoNgcjFLl1x7cAHV6Lb2UQQe+K8fPYVQ5XNJT2fjjFb7mTIlerxFgERbYtxbp5GywdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = '19ab685c8fe84c2e4ec1bc036597bbec'
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route('/line', methods=['GET'])
def line():
    line_bot_api.push_message(
            'Ud1aab8bcda543156713b1a9946fefc79',
            TextSendMessage(text='hello')
        )
    return 'sent'

@app.route('/linesend/<message>', methods=['GET'])
def linesend(message):
    line_bot_api.push_message(
            'Ud1aab8bcda543156713b1a9946fefc79',
            TextSendMessage(text=message)
        )
    return '{}\nsent'.format(message)

@app.route('/linepost', methods=['POST'])
def linepost():
    data = request.get_json()
    pitch = data['pitch']
    roll = data['roll']
    message = 'pitch: {}\nroll: {}'.format(pitch, roll)
    line_bot_api.push_message(
            'Ud1aab8bcda543156713b1a9946fefc79',
            TextSendMessage(text=message)
        )
    res = {'pitch':pitch, 'roll':roll}
    return make_response(jsonify(res))

@app.route('/receivepost', methods=['POST'])
def receivepost():
    json = request.get_json()
    data = json['data']
    return make_response(jsonify(data))


from flask import render_template
@app.route('/map', methods=['GET'])
def map():
    lat = 36.00
    lng = 140.00
    return render_template('test.html', lat=lat, lng=lng)

if __name__=='__main__':
    # app.run(debug=True)
    port = os.getenv('PORT')
    app.run(host='0.0.0.0', port=port)

