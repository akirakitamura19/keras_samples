
from __future__ import print_function
import re
import glob
import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.applications.xception import Xception
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    file_path_list = sorted(
        glob.glob(os.getenv("HOME") + "/data/toray/typeD/Ibutu_kizu/*/*.JPG"))

    X = [np.asarray(Image.open(file_path).resize((250, 250)))
         for file_path in file_path_list]

    labels = [os.path.split(os.path.split(file_path)[0])[1]
              for file_path in file_path_list]
    categories = sorted(list(set(labels)))
    Y = [categories.index(label) for label in labels]

    # list => numpy
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y)

    X /= 255

    X_train = np.concatenate((X[np.where(Y == 0)][100:288],
                              X[np.where(Y == 1)][100:288],
                              X[np.where(Y == 2)][100:288],
                              X[np.where(Y == 3)][100:288],
                              X[np.where(Y == 4)][100:288]), axis=0)
    X_test = np.concatenate((X[np.where(Y == 0)][:100],
                             X[np.where(Y == 1)][:100],
                             X[np.where(Y == 2)][:100],
                             X[np.where(Y == 3)][:100],
                             X[np.where(Y == 4)][:100]), axis=0)

    Y_train = np.concatenate((Y[np.where(Y == 0)][100:288],
                              Y[np.where(Y == 1)][100:288],
                              Y[np.where(Y == 2)][100:288],
                              Y[np.where(Y == 3)][100:288],
                              Y[np.where(Y == 4)][100:288]), axis=0)
    Y_test = np.concatenate((Y[np.where(Y == 0)][:100],
                             Y[np.where(Y == 1)][:100],
                             Y[np.where(Y == 2)][:100],
                             Y[np.where(Y == 3)][:100],
                             Y[np.where(Y == 4)][:100]), axis=0)

    # one-hot
    Y_train = np_utils.to_categorical(Y_train, 5)
    Y_test = np_utils.to_categorical(Y_test, 5)

    # Shuffle train dataset(important)
    X_train, Y_train = shuffle(X_train, Y_train)

    # Show shape
    print("X_train.shape: {0}".format(X_train.shape))
    print("Y_train.shape: {0}".format(Y_train.shape))
    print("X_test.shape: {0}".format(X_test.shape))
    print("Y_test.shape: {0}".format(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):

    # xception model without top
    transfer_model = Xception(include_top=False,
                              weights='imagenet',
                              input_shape=(250, 250, 3))

    # Adding custom Layers
    x = transfer_model.output
    x = Flatten()(x)
    x = Dense({{choice([32, 64, 128])}}, activation="relu")(x)
    x = Dropout({{uniform(0.5, 1)}})(x)
    predictions = Dense(5, activation="softmax", name='prediction')(x)

    # creating the final model
    model = Model(inputs=transfer_model.input, outputs=predictions)

    for layer in model.layers[:-18]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr={{uniform(1e-4, 1e-2)}},
                                decay={{uniform(1e-7, 1e-5)}},
                                momentum={{uniform(0.7, 0.9)}},
                                nesterov=True),
                  metrics=['accuracy'])

    # Early stop function
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    # Train model
    history = model.fit(X_train, Y_train,
                        batch_size={{choice([32, 64])}},
                        epochs=100,
                        validation_split=0.25,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=1)

    # Evaluate by accuracy
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': score, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    # TPE algorithm for search better hyper-parameter
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=Trials())

    # Load and preprocess dataset
    X_train, Y_train, X_test, Y_test = data()

    # Evaluate best model accuracy&loss and hyper-parameter
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    # Evaluate confusion-matrix and f-score
    y_pred = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred, -1)
    y_true = np.argmax(Y_test, -1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Save best model
    model_json_str = best_model.to_json()
    open('best_model_weights/Xception_model.json', 'w+').write(model_json_str)
    best_model.save_weights('best_model_weights/Xception_weights.h5')
