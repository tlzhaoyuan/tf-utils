import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from dataset.data_generator import DataGenerator, DataGeneratorV2
from embedding import tokenize, encode_label
from model.text_classification_cnn import TextClassificationCNNModelV2
from util import train_val_test_split

SEED = 1234
SHUFFLE = True
DATA_FILE = 'data/news.csv'
INPUT_FEATURE = 'title'
OUTPUT_FEATURE = 'category'
FILTERS = "!\"'#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
LOWER = True
CHAR_LEVEL = True
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_FILTERS = 50
FILTER_SIZE = 3 # tri-grams
HIDDEN_DIM = 100
DROPOUT_P = 0.1
LEARNING_RATE = 1e-3
EARLY_STOPPING_CRITERIA = 3

def main():
    df = pd.read_csv(DATA_FILE)
    X = df[INPUT_FEATURE].values
    y = df[OUTPUT_FEATURE].values

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        shuffle=SHUFFLE
    )

    X_train, X_val, X_test, vocab_size = tokenize(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        char_level=CHAR_LEVEL,
        lower=LOWER,
        filters=FILTERS,
        one_hot=True
    )
    y_train, y_val, y_test, classes, class_weights = encode_label(
        y_train=y_train,
        y_test=y_test,
        y_val=y_val
    )

    train_generator = DataGeneratorV2(
        X=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size,
        max_filter_size=FILTER_SIZE
    )
    val_generator = DataGeneratorV2(
        X=X_val,
        y=y_val,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size,
        max_filter_size=FILTER_SIZE
    )
    test_generator = DataGeneratorV2(
        X=X_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size,
        max_filter_size=FILTER_SIZE
    )

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_CRITERIA, verbose=1, mode='min'),
                 ReduceLROnPlateau(patience=1, factor=0.1, verbose=0),
                 TensorBoard(log_dir='tensorboard', histogram_freq=1, update_freq='epoch')]
    model = TextClassificationCNNModelV2(
        dropout_p=DROPOUT_P,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(classes),
        filter_size=FILTER_SIZE,
        num_filters=NUM_FILTERS
    )
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(
        train_generator,
        callbacks=callbacks,
        validation_data=val_generator,
        epochs=NUM_EPOCHS,
        class_weight=class_weights,
        shuffle=False
    )

if __name__ == '__main__':
    main()