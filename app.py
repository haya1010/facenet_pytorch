from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import requests
import io
from flask import Flask, jsonify, abort, make_response, request
import os

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
        urls[key]
        res[key] = {}
        for url in urls[key]:
            img1 = feature_vector(url)
            img2 = feature_vector(compare)
            res[key][url] = str(cosine_similarity(img1, img2))
        return res

def get_result(compare, urls):
  tmp = {}
  for url in urls:
    img1 = feature_vector(url)
    img2 = feature_vector(compare)
    tmp[url] = cosine_similarity(img1, img2)
  print(tmp)
  return tmp


app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello facenet_pytorch'

@app.route('/test')
def test():
    url1 = 'https://thetv.jp/i/nw/1023326/10207156.jpg?w=615'
    url2 = 'https://cdn.hinatazaka46.com/files/14/diary/official/member/moblog/202012/mobMCWhSw.jpg'
    url3 = 'https://coconutsjapan.com/wp-content/uploads/2020/06/jiji-kawatahina-hinatazaka46.jpg'
    url4 = 'https://thetv.jp/i/nw/1023326/10207156.jpg?w=615'

    img1 = feature_vector(url1)
    img2 = feature_vector(url2)
    result = cosine_similarity(img1, img2)
    res = {}
    res['result'] = str(result)
    return make_response(jsonify(res))

@app.route('/post')
def post():
    data = request.get_json()
    compare = data['compare']
    urls = data['urls']
    res = make_result(compare, urls)
    return make_response(jsonify(res))



if __name__=='__main__':
    # app.run(debug=True)
    port = os.getenv('PORT')
    app.run(host='0.0.0.0', port=port)

