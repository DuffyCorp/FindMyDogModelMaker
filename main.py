from tensorflow import keras
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import ImageFile, Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import itertools

ImageFile.LOAD_TRUNCATED_IMAGES = True

VERSION = 2

image_size = 224

model_file_name = f'output/v{VERSION}/model.h5'

# Load the labels
class_names = open("data/labels.txt", "r").readlines()

n_classes = len(class_names)

def printAccuracyGraph(epochs, acc, val_acc):
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(f'output/v{VERSION}/AccuracyGraph.png')
    plt.clf()

def printLossGraph(epochs, loss, val_loss):
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(f'output/v{VERSION}/LossGraph.png')
    plt.clf()

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(200,200))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'output/v{VERSION}/ConfusionGraph.png')

if not os.path.exists(model_file_name):
    mobile_net_v2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    # Freeze the layers except the last 8 layers
    for layer in mobile_net_v2.layers[:-8]:
        layer.trainable = False
    # Create the model
    model = models.Sequential(name = "Dog_Breed_Classification")

    # Add the vgg convolutional base model
    model.add(mobile_net_v2)

    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu')) # dense layer 2
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu')) # dense layer 3
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax')) # final layer with softmax activation

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Change the batchsize according to your system RAM
    train_batchsize = 100
    val_batchsize = 10
    image_size = 224

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        'data/valid',
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1)
  
  

    # Save the model
    model.save(model_file_name)
    with open(f'output/v{VERSION}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Show final result
    history.history['accuracy'][-1]

    # Plot the results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


model = load_model(f"output/v{VERSION}/model.h5", compile=False)

with open(f'output/v{VERSION}/trainHistoryDict', 'rb') as handle:
    history = pickle.load(handle)


model.summary()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("test/dog.png").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

test_batch_size = 1
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory='data/test',
    target_size=(image_size, image_size),
    batch_size=test_batch_size,
    class_mode='categorical',
    shuffle=False
)

if not os.path.exists(f'output/v{VERSION}/AccuracyGraph.png'):
    print("Accuracy graph")

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    epochs = range(len(acc))

    printAccuracyGraph(epochs, acc, val_acc)

if not os.path.exists(f'output/v{VERSION}/LossGraph.png'):
    print("Loss graph")

    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(loss))

    printLossGraph(epochs, loss, val_loss)


if not os.path.exists(f'output/v{VERSION}/ConfusionGraph.png'):
    print('Confusion Matrix')

    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(test_generator.classes, y_pred)

    plot_confusion_matrix(cm, class_names, title='Confusion Matrix')


if not os.path.exists(f"output/v{VERSION}/ClassificationResults.csv"):
    print("Classification Report")

    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    report = classification_report(test_generator.classes, y_pred, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df.to_csv(f"output/v{VERSION}/ClassificationResults.csv", encoding='utf-8', index=False)


if not os.path.exists(f"output/v{VERSION}/TFLITE/model.tflite"):
    if not os.path.exists(f"output/v{VERSION}/TFLITE"):
        os.makedirs(f"output/v{VERSION}/TFLITE")
    print("Creating TFLite model")

    TF_LITE_MODEL_FILE_NAME = f"output/v{VERSION}/TFLITE/model.tflite"

    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = tf_lite_converter.convert()

    tflite_model_name = TF_LITE_MODEL_FILE_NAME

    open(tflite_model_name, "wb").write(tflite_model)