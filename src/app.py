import argparse
from flask import Flask, request, jsonify
from flask_dropzone import Dropzone
import os
import os.path as osp
import json
from API import predict
from classifier import Classifier


app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=1,
    DROPZONE_MAX_FILES=1000,
)
if not os.path.exists(app.config['UPLOADED_PATH']):
    os.makedirs(app.config['UPLOADED_PATH'])

dropzone = Dropzone(app)

def init_folders():
    import os, errno
    folders = [
        'resources/mid',
        'resources/model',
        'resources/mxl',
        'resources/sheets',
        'resources/sounds',
        'resources/templates',
        'uploads',
        'cached_segmenter',
        'cached_stream'
    ]
    for f in folders:
        try:
            os.makedirs(f)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

init_folders()

def get_model():
    import urllib
    model = urllib.URLopener()
    model.retrieve("https://www.dropbox.com/s/vllgec938hdxrcq/keras_modelv3.h5?dl=1", "resources/model/keras_modelv3.h5")

get_model()

# TODO: do we really need CORS?

from music21 import vexflow, note

FOLDER_MXL = 'resources/mxl'
EXTENSION_MXL = '.musicxml'


@app.route("/")
def home():
    return "Hello world!"


# retrieves a sample music xml that is stored in /mxl
@app.route("/get_music_xml")
def get_music_xml():
    sheet = request.args['sheet_name']
    fn = osp.join(FOLDER_MXL, sheet)
    fn = fn + EXTENSION_MXL if ~fn.endswith(EXTENSION_MXL) else fn
    handle = open(fn, 'r', encoding='utf-8')
    data = handle.read().replace('\n', '')
    handle.close()

    return jsonify(data)


# uploads an image to /uploads
@app.route("/image_upload", methods=['GET', 'POST'])
def image_upload():
    print('received file')
    if request.method == 'POST':
        f = request.files.get('file')
        fp = os.path.join(app.config['UPLOADED_PATH'], f.filename)
        f.save(fp)
        print('saved file. predicting...')
        global classifier
        xml, b64 = predict(fp, classifier)
        print('done predicting. musicxml is sending...')
        return json.dumps({'success': True, "data": {"xml": xml, "midi": str(b64)[2:-1]}}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'success': False}), 200, {'ContentType': 'application/json'}


@app.after_request
def add_ua_compat(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers['Access-Control-Allow-Headers'] = 'Cache-Control, X-Requested-With'
    return response


@app.before_first_request
def init_model():
    global classifier
    classifier = Classifier()


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store_true', help='debug the flask app')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(debug=args.d)
