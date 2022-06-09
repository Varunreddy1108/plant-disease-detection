import base64
from typing import Any
from flask import Flask, render_template, request, send_file, redirect, url_for
import string
import random
import os
import cv2
import sqlite3
from tensorflow.keras.models import load_model
import disease_detector as dd

upload_folder = os.getcwd()

conn = sqlite3.connect('crop_iden.db')
cursor = conn.cursor()
cursor.execute(
    'create table if not exists crop_table (id integer primary key autoincrement, lat_log varchar, image_name varchar, plant_name varchar, description text)')
conn.commit()
conn.close()


def name_gen() -> str:
    char = string.ascii_letters
    return ''.join(random.choices(char, k=6))


app: Flask = Flask(__name__)


@app.route('/')
def hello_world() -> Any:
    msg = request.args.get('msg')
    if msg:
        print(msg)
        return render_template('home.html', message=msg)
    return render_template('home.html')


@app.route('/<lat>/<lon>', methods=['POST'])
def get_data(lat: str, lon: str) -> str:
    print(request.form)
    a = request.files
    file = a['image']
    print(file.filename)
    filename = name_gen()
    with sqlite3.connect('crop_iden.db') as db:
        cursor = db.cursor()
        cursor.execute('insert into crop_table (lat_log, image_name) values(?,?)', [f'{lat},{lon}', filename])
        db.commit()
    print(filename)
    file.save(os.path.join(upload_folder, filename + '.png'))
    return f'{filename}'


@app.route('/img_prev/<img_name>')
def get_image(img_name: str) -> Any:
    return send_file(path_or_file=f'{img_name}.png', mimetype='image/png')


@app.route('/detail/<img_name>')
def image_preview(img_name: str) -> Any:
    res,perc = dd.predictor(f'{img_name}.png')
    print("percentage========> ",perc)
    if perc > 99.99999 :
        return redirect(url_for('hello_world', msg="please retake image"))

    with sqlite3.connect('crop_iden.db') as db:
        cursor = db.cursor()
        cursor.execute('update crop_table set plant_name = ?, description = ? where  image_name=?',
                       [list(res.keys())[0], list(res.values())[0], img_name])
        loc = cursor.execute('select lat_log from crop_table where image_name = ?', [img_name]).fetchone()
        db.commit()
    return render_template('detail.html', context={'filename': img_name, 'predicted_data': res, 'location': loc[0]})


if __name__ == '__main__':
    app.run(ssl_context='adhoc', host='0.0.0.0', debug=True)
    # help(send_file)

