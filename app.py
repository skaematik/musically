import argparse
from flask import Flask, request, jsonify
from flask_dropzone import Dropzone
import os
import os.path as osp
import json
from API import predict

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

# TODO: do we really need CORS?

from music21 import vexflow, note

FOLDER_MXL = 'mxl'
EXTENSION_MXL = '.musicxml'


@app.route("/")
def home():
    return "Hello world!"


# @app.route("/musically")
# def musically():
#     # text = request.args.get('query', '')

#     print(request)

#     # will call the actual function here

#     # dummy testing
#     n = note.Note('C#4')
#     to_be_sent = vexflow.toMusic21j.fromObject(n, mode='html')
#     print(to_be_sent)

#     return jsonify({
#         'to_be_sent': to_be_sent
#     })


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
        xml, b64 = predict(fp)

        print('done predicting. musicxml is sending...')
        return json.dumps({'success': True, "data": {"xml": xml, "midi": str(b64)[2:-1] }}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'success': False}), 200, {'ContentType': 'application/json'}
    # return render_template('index.html')


@app.after_request
def add_ua_compat(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers['Access-Control-Allow-Headers'] = 'Cache-Control, X-Requested-With'
    return response


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store_true', help='debug the flask app')
    # parser.add_argument('--port', type=int, default=5000,
    #                     help='the port for the app to run on')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(debug=args.d)
