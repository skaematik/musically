# musically
COMP9517 Project - Optical Music Recognition

## contributors
[@Angus-fletcher](https://github.com/Angus-fletcher) (Angus Fletcher)<br>
[@skaematik](https://github.com/skaematik) (Annie Zhang)<br>
[@YiweiHan](https://github.com/YiweiHan) (Yiwei Han)

## dependencies
Install dependencies with `pip install -r requirements.txt`
- python 3.6
- opencv contrib 3.4.012
- numpy 1.14.2

## development dependencies
See `development-requirements.txt`. Only needed for development/testing/visualisation.

## how to set up a virtualenv
Windows:

```bash
/path/to/python/Scripts/virtualenv.exe --python=/path/to/python3/python.exe venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Other:

```bash
virtualenv --python=/path/to/python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## for the server

The start the development server:

```bash
python app.py
```

The following flags are available:

```
-d (debug mode)
```

### for the client

To start the React app:

```
cd client
yarn
yarn start
```