# https://www.pluralsight.com/resources/blog/cloud/create-a-serverless-python-api-with-aws-amplify-and-flask
# https://flask.palletsprojects.com/en/3.0.x/patterns/fileuploads/

import json
import os
import base64
import awsgi
import boto3
from flask import Flask, flash, render_template, request
from werkzeug.utils import secure_filename
import subprocess

import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
lambda_client = boto3.client('lambda', region_name=os.environ['AWS_REGION'])
s3_client = boto3.client('s3', region_name=os.environ['AWS_REGION'])
inference_function = os.environ["BIRDSONG_FUNCTION"]
audio_bucket = os.environ["AUDIO_BUCKET"]

def handler(event, context):
    # https://github.com/slank/awsgi/issues/73
    if 'httpMethod' not in event:
        event['httpMethod'] = event['requestContext']['http']['method']
    if 'path' not in event:
        event['path'] = event['requestContext']['http']['path']
    if 'queryStringParameters' not in event:
        event['queryStringParameters'] = {}
    app.logger.info(event['headers'])
    return awsgi.response(app, event, context)

def get_response(fname):
    with open(fname, 'rb') as f:
        payload_body = base64.b64encode(f.read()).decode('utf-8')
    response = lambda_client.invoke(FunctionName=inference_function, Payload=json.dumps({'body': payload_body }))
    response = json.loads(response['Payload'].read())
    return response

@app.route("/", methods=['POST'])
def upload_file():
    print("FILES %s",request.files)
    print("DATA %d",len(request.data))
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No filename')
            return {"code": 500,
                    "description": "No filename"}
        else:
            sec_filename = secure_filename(file.filename)
            local_filename = os.path.join(app.config['UPLOAD_FOLDER'], sec_filename)
            extract_filename = local_filename.rsplit(".", 1)[0] + "_FFMPEG.ogg"
            print("SEC_FILENAME", local_filename)
            file.save(local_filename)
    else:
        flash('No file')
        return {"code": 500,
                "description": "No file"}

    put_response = s3_client.upload_file(local_filename, audio_bucket, sec_filename)
    print("UPLOAD_FILE", put_response)
    response = get_response(local_filename)
    print("RESPONSE", response)
    if response['code'] == 500:
        print("Could not process original format")
        print("Converting to OGG")
        # https://github.com/riad-azz/audio-extract/blob/master/audio_extract/ffmpeg.py
        # https://apple.stackexchange.com/questions/13221/whats-the-best-way-to-extract-audio-from-a-video-file
        # subprocess.run(['ls','-l','/tmp'])
        if os.path.exists(extract_filename):
            os.remove(extract_filename)
        subprocess.run(['ffmpeg', '-v', 'warning', '-i', local_filename, '-vn', '-acodec', 'libvorbis', extract_filename,])
        response = get_response(extract_filename)
    return response['top_results']

@app.route("/", methods=['GET'])
def request_file():
    return render_template('get.html')
