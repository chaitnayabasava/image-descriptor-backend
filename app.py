from flask import request, Flask
from flask import jsonify
from PIL import Image
import io, os

from predict import get_predictions

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

@app.route('/', methods=['GET'])
def check_status():
    return jsonify("Running...")

@app.route('/image-upload', methods=['POST'])
def image_upload():
    r = request.args
    img = Image.open(io.BytesIO(request.files['file'].read())).convert("RGB")
    sentences = get_predictions(img, r['model_name'], int(r['beam_size']))
    return jsonify({'sentences': sentences})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get("PORT", 5000))