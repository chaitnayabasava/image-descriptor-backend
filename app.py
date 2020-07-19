from flask import request, Flask
from flask import jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import io

from predict import get_predictions

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/predict_petal_length": {"origins": "*"}})

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response

@app.route('/image-upload', methods=['POST'])
@cross_origin(origin='*',headers=['Content- Type'])
def image_upload():
    r = request.args
    img = Image.open(io.BytesIO(request.files['file'].read())).convert("RGB")
    sentences = get_predictions(img, r['model_name'], int(r['beam_size']))
    return jsonify({'sentences': sentences})

if __name__ == '__main__':
    app.run(host='0.0.0.0')