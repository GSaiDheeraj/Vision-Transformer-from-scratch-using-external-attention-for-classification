from flask import Flask,render_template, url_for , redirect
#from forms import RegistrationForm, LoginForm
#from sklearn.externals import joblib
from flask import request
import numpy as np
from PIL import Image
from flask import flash
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck


import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf

import keras
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K

from keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')


app.config['SECRET_KEY'] = "UddA58IkCqP5nZkwEzA7YA"



dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'




num_classes = 7
input_shape = (100, 100, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 100  # We'll resize input images to this size
patch_size = 7  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 14
mlp_head_units = [2048, 1024]  #

x_train = np.load("C:\\Users\\saidh\\Desktop\\Ocular Disease Detection\\data\\image_data_50.npy")
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

model1 = create_vit_classifier()






























# global graph
# graph = tf.get_default_graph()
# model1 = tensorflow.keras.models.load_model("eyedisease.h5")
# model1 = tensorflow.keras.Model()
# model1.build(input_shape = (50,50))
# model1.built = True
model1.load_weights("disease.h5")

#pneumonia
def api1(full_path):
    #with graph.as_default():
    data = keras.preprocessing.image.load_img(full_path, target_size=(100, 100, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0/ 255
    predicted = model1.predict(data)
    return predicted

#Pneumonia
@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():
    #with graph.as_default():
    if request.method == 'GET':
        return render_template('malaria.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'AMD', 2: 'Cataract', 3: 'Diabetes', 4: 'Glaucoma', 5: 'Myopia',6: 'Hypertension'}
            result = api1(full_name)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            if accuracy < 85:
                prediction = "Please, Check with the Doctor."
            else:
                prediction = "Result is accurate"

            return render_template('malariapredict.html', image_file_name=file.filename, label=label, accuracy=accuracy,
                                   prediction=prediction)
        except:
            flash("Please select the Cell image first !!", "danger")
            return redirect(url_for("Malaria"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

#logged in Home page
@app.route("/")
@app.route("/home")
def index1():
    return render_template("home.html")

@app.route("/about")
def index2():
    return render_template("about.html")

@app.route("/Malaria")
def Pneumonia():
    return render_template("malaria.html")


if __name__ == "__main__":
	app.run(debug=True)
