# https://www.pluralsight.com/resources/blog/cloud/create-a-serverless-python-api-with-aws-amplify-and-flask
# https://flask.palletsprojects.com/en/3.0.x/patterns/fileuploads/

import json
import os
import base64
import requests
import awsgi
import boto3
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename

import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
client = boto3.client('lambda', region_name=os.environ['AWS_REGION'])
inference_function = os.environ["BIRDSONG_FUNCTION"]

def handler(event, context):
    print(event['headers'])
    print(event['multiValueHeaders'])
    return awsgi.response(app, event, context)

@app.route("/", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    else:
        file = request.files['file']
        if file.filename == '':
            flash('Empty filename')
            return redirect(request.url)
        else:
            sec_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save( sec_filename)
            print("SEC_FILENAME", sec_filename)
            with open(sec_filename, 'rb') as f:
                payload_body = base64.b64encode(f.read()).decode('utf-8')
            response = client.invoke(FunctionName=inference_function, Payload=json.dumps({'body': payload_body }))
            response = json.loads(response['Payload'].read())
            top_results = response['top_results']
            if len(top_results) > 0:
                result_str = "Code for top bird" if len(top_results) == 1 else "Codes for top birds"
                current_result = [f"{result_str} in {file.filename}:"] + [f"<li><a href=\"https://ebird.org/species/{bird_code}\"> {bird_code}</a>: {bird_score:.1f}</li>" for bird_code, bird_score in top_results]
            else:
                current_result = [f"No birds found in {file.filename}."]
            return request_file("<p>"+"".join(current_result)+"</p>")

@app.route("/", methods=['GET'])
def request_file(prev_result=''):
    return f'''
    <!doctype html>
    <p>Upload audio file</p>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    {prev_result}
    '''

