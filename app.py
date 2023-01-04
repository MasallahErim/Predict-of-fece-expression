from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import sqlalchemy
import pandas as pd

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

import os
import numpy as np
import matplotlib.pyplot as plt





app = Flask(__name__)
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///c:\\Users\\Maşallah\\Desktop\\Deeplearning\\flask\\uygulama\\app.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app=app)

# class Upload(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.Text, unique=True, nullable=False)
#     lastname = db.Column(db.Text, unique=True, nullable=False)

# with app.app_context():
#     db.create_all()

# import tensorflow as tf
# graph = tf.get_default_graph()




UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/face_expression', methods=["GET", "POST"])
def face_expression():
    return render_template("face_expression.html")




def image_process(img_path, model_path):
    model = load_model(model_path)
    img = load_img(img_path,target_size=(48,48))
    img=np.expand_dims(img,axis=0)
    img = img/ 255
    img_predict = model.predict(img)
    img_predict = img_predict.round()
    global predict_expression 

    if img_predict[0][0]>0:
        predict_expression='Kızgınsınız'
    elif img_predict[0][1]>0:
        predict_expression='İğreniyorsunuz'
    elif img_predict[0][2]>0:
        predict_expression='korkuyorsunuz'
    elif img_predict[0][3]>0:
        predict_expression='Mutlusunuz'
    elif img_predict[0][4]>0:
        predict_expression='Üzgünsünüz'
    elif img_predict[0][5]>0:
        predict_expression='Şaşkınsınız' 
    else:
        predict_expression='Doğalsınız'
        print("Congratulations you are nothing")
    
    
    return predict_expression



def sql_operations():

    dbEngine=sqlalchemy.create_engine('sqlite:///c:\\Users\\Maşallah\\Desktop\\Deeplearning\\flask\\uygulama\\movie_datadb.db')
    df=  pd.read_sql('select * from Movies',dbEngine)
    df = df.sample(n=1).to_numpy()
    return df[0][0], df[0][1], df[0][2],df[0][3], df[0][4]
    




@app.route('/face_expression/add', methods=['POST', "GET"])
def upload_image():
    
    if request.method =="POST" :
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('Resim başarıyla yüklendi ve aşağıda gösterildi <-----> Image successfully uploaded and displayed below')
            name = request.form.get("firstname")
            lastname = request.form.get("lastname")
            
            img_path = "./static/uploads/"+filename
            model_path = "./CNN_Model1"
            global image_process
            predict = image_process(img_path,model_path)
            movie_name, movie_year,movie_genre, movie_rate, movie_summary = sql_operations()
            
            return render_template('face_expression.html', filename=filename,
                                                            name = name,
                                                            lastname = lastname,
                                                            predict = predict,
                                                            movie_name = movie_name,
                                                            movie_year = movie_year,
                                                            movie_genre = movie_genre,
                                                            movie_rate = movie_rate,
                                                            movie_summary = movie_summary )
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return redirect(url_for("face_expression"))





@app.route('/face_expression/<filename>')
def display_image(filename):
    return redirect(url_for('static', 
                            filename='uploads/' + filename),
                            code=301)



if __name__ == "__main__":
    app.run(debug=True)


