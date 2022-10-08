from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pickle
import tensorflow
import keras
from keras.utils import load_img, img_to_array
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embedded.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])



def feature_extraction(img_path,model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    print(indices)
    return indices


app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods = ['post'])
def imageWork():
    imagefile = request.files['image_input']
    imagePath = "./static/uploads/" + imagefile.filename
    imagefile.save(imagePath) 
    image_sample = feature_extraction(imagePath,model)
    indices = recommend(image_sample,feature_list)
    list_add = []
    for file in indices[0][0:8]:
        list_add.append(filenames[file])
    list_add
    display_image = list_add[0]
    return render_template('index.html', imagelist = list_add, display_image = display_image)


if __name__ == "__main__":
    app.run(debug=True)