import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
# TODO: do we really need CORS?
CORS(app)


@app.route("/")
def home():
    return "Hello world!"


@app.route("/musically")
def musically():
    text = request.args.get('query', '')
    # will call function here
    return jsonify({
        'text': text
    })


@app.after_request
def add_ua_compat(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store_true', help='debug the flask app')
    parser.add_argument('--port', type=int, default=5000,
                        help='the port for the app to run on')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    app.run(port=args.port, debug=args.d)
