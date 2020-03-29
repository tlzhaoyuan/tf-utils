import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# Arguments
from dataset.data_generator import DataGenerator
from embedding import encode_label
from model import MLP
from util import train_val_test_split, standardize

SEED = 1234
DATA_FILE = "data/spiral.csv"
SHUFFLE = True
MODEL_FILE = 'model.hdf5'
INPUT_DIM = 2
NUM_CLASSES = 3
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_DIM = 100
DROPOUT_P = 0.1
LEARNING_RATE = 1e-2
EARLY_STOPPING_CRITERIA = 3 # deteriorating epochs

# Set seed for reproducability
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    # Load data
    df = pd.read_csv(DATA_FILE, header=0)
    X = df[['X1', 'X2']].values
    y = df['color'].values
    df.head(5)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y,
                                                                          test_size=TEST_SIZE,
                                                                          val_size=VAL_SIZE,
                                                                          shuffle=SHUFFLE)

    X_train, X_val, X_test = standardize(X_train, X_val, X_test)
    y_train, y_val, y_test, _, class_weights = encode_label(y_train, y_val, y_test)

    # Dataset generator
    training_generator = DataGenerator(X=X_train,
                                       y=y_train,
                                       batch_size=BATCH_SIZE,
                                       shuffle=SHUFFLE)
    validation_generator = DataGenerator(X=X_val,
                                         y=y_val,
                                         batch_size=BATCH_SIZE,
                                         shuffle=SHUFFLE)
    testing_generator = DataGenerator(X=X_test,
                                      y=y_test,
                                      batch_size=BATCH_SIZE,
                                      shuffle=SHUFFLE)
    model = MLP(hidden_size=HIDDEN_DIM,
                dropout_p=DROPOUT_P,
                num_classes=NUM_CLASSES)
    model.sample(input_shape=(INPUT_DIM,))
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    run_dir = 'tensorboard/callback_run'
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOPPING_CRITERIA, verbose=0),
                 ModelCheckpoint(filepath=MODEL_FILE, monitor='val_loss', mode='min', verbose=0, save_best_only=True,
                                 save_weights_only=True),
                 ReduceLROnPlateau(monitor='val_loss', mode='min', patience=1, factor=0.1, verbose=0),
                 TensorBoard(log_dir=run_dir, histogram_freq=1, update_freq='epoch')]
    # Training
    model.fit_generator(generator=training_generator,
                        epochs=NUM_EPOCHS,
                        validation_data=validation_generator,
                        shuffle=False,
                        class_weight=class_weights,
                        verbose=1,
                        callbacks=callbacks)  # add callbacks