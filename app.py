import flask
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from os import path as osp
import boto3
from werkzeug.utils import secure_filename
import json

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

access_key = "AKIAQMFRBMFSAPYOR2ZJ"
secret_access_key = "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"
bucket = 'archi-ai'

s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_access_key,
        region_name='ap-northeast-2'
    )

# model download
def get_file_from_s3(s3, key, model_name):
    print(f'Try to get from "{key}"')
    try:
        dest = osp.join(os.getcwd(), model_name)
        #os.makedirs(osp.dirname(dest), exist_ok=True)
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=dest
        )
        print('success!\n')
    except:
        print(f'Failed to get {key}')
        return False

@app.route('/')
def index():
    return flask.render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# cache 사용 안함
@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r

# 예측
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        class_check = request.form.get('model_class')
        model = models[class_check]

        with open('class_list_'+ model_and_ds[class_check] + '.txt') as f:
            lines = f.readlines()
            class_list = list(map(lambda s: s.strip(), lines))
            f.close()

        uploaded = False
        file = request.files['image']

        if not file: 
            return render_template('index.html', label="No Files")
        else :
            uploaded=True

        if file and allowed_file(file.filename) :
            #filename = secure_filename(file.filename)
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            abs_filepath = os.path.join('static/',filepath)
            file.save(abs_filepath)
        else :
            return render_template('index.html',label='not allowed file format')

        
        img = tf.io.read_file(abs_filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        pix = np.array(img)
        pix = pix.reshape((1,) + pix.shape)

        prediction = model.predict(pix)
        prediction = prediction[0]

        top2_idx = prediction.argsort()
        top2 = {}

        for i,v in enumerate(class_list):
            if i == top2_idx[-1] or i == top2_idx[-2] :
                top2[v] = round(prediction[i]*100,2)
        top2 = sorted(top2.items(), key=(lambda x:x[1]),reverse=True)
        
        # description of top1
        if(class_check == 'class5'):
            for i, v in top2 :
                top1_class = i
                top1_img = descriptions[class_check][i]['img_src']
                top1_des = descriptions[class_check][i]['detail']
                break
        else:#아직 class9에 대한 description 없음
            top1_class = ""
            top1_img = ""
            top1_des = ""
            
        return render_template('index.html', uploaded=uploaded, filename=filename, filepath=filepath, top2=top2,
                    top1_class=top1_class, top1_img=top1_img, top1_des=top1_des)



if __name__ == '__main__':
    #model download
    model_and_ds = {
            'class5' : 'ver2', 
            'class9' : 'ver3'
        }
    
    models = {}
    
    for model_ver, dataset_ver in model_and_ds.items():
        model_name = 'model_' + model_ver + '.h5'
        model_path = osp.join(os.getcwd(), model_name)
        
        if osp.exists(model_path) == False:
            key = osp.join('style_analysis/style_image_classification/room_style_classifier_'+ model_name)
            value = get_file_from_s3(s3, key, model_name)
        
        models[model_ver] = keras.models.load_model(model_name)
    
    with open("static/style_descriptions/style_descriptions.json", "r", encoding='utf-8') as f: 
        data = f.read() 
        descriptions = json.loads(data) 

    os.makedirs(osp.join(os.getcwd(),"static/uploads"), exist_ok=True)

    app.run(debug=True)