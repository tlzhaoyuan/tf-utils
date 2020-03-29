import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau, EarlyStopping
# Arguments
from tensorflow.keras import Input
from tensorflow.keras.layers import SimpleRNN
from dataset.data_generator import DataGenerator
from embedding import tokenize, encode_label
from model import ModelToInspect
from model.text_classification_rnn import TextClassificationRNNModel
from util import train_val_test_split

SEED = 1234
SHUFFLE = True
DATA_FILE = 'data/news.csv'
INPUT_FEATURE = 'title'
OUTPUT_FEATURE = 'category'
FILTERS = "!\"'#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
LOWER = True
CHAR_LEVEL = False
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_EPOCHS = 10
BATCH_SIZE = 256
EMBEDDING_DIM = 100
RNN_HIDDEN_DIM = 128
RNN_DROPOUT_P = 0.1
NUM_LAYERS = 1
HIDDEN_DIM = 100
DROPOUT_P = 0.1
LEARNING_RATE = 1e-3
EARLY_STOPPING_CRITERIA = 3
# Set seed for reproducability
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[INPUT_FEATURE].values
    y = df[OUTPUT_FEATURE].values

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y,
                                                                          test_size=TEST_SIZE,
                                                                          val_size=VAL_SIZE,
                                                                          shuffle=SHUFFLE)

    X_train, X_val, X_test, vocab_size = tokenize(X_train=X_train,
                                      X_val=X_val,
                                      X_test=X_test,
                                      filters=FILTERS,
                                      lower=LOWER,
                                      char_level=CHAR_LEVEL)
    y_train, y_val, y_test, classes, class_weights = encode_label(y_train=y_train,
                                          y_val=y_val,
                                          y_test=y_test)
    # Dataset generator
    training_generator = DataGenerator(X=X_train,
                                       y=y_train,
                                       batch_size=BATCH_SIZE,
                                       shuffle=SHUFFLE)
    validation_generator = DataGenerator(X=X_val,
                                         y=y_val,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False)
    testing_generator = DataGenerator(X=X_test,
                                      y=y_test,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False)

    # Input
    sequence_size = 8  # words per input
    x = Input(shape=(sequence_size, EMBEDDING_DIM))
    # RNN forward pass (many to one)
    rnn = SimpleRNN(units=RNN_HIDDEN_DIM,
                    dropout=DROPOUT_P,
                    recurrent_dropout=RNN_DROPOUT_P,
                    return_sequences=False,  # only get the output from the last sequential input
                    return_state=True)
    output, hidden_state = rnn(x)

    # RNN forward pass (many to many)
    rnn = SimpleRNN(units=RNN_HIDDEN_DIM,
                    dropout=DROPOUT_P,
                    recurrent_dropout=RNN_DROPOUT_P,
                    return_sequences=True,  # get outputs from every item in sequential input
                    return_state=True)
    outputs, hidden_state = rnn(x)

    # Get the first data point
    sample_inputs, sample_y = training_generator.create_batch([0])

    sample_X, sample_seq_length = sample_inputs

    # RNN cell
    simple_rnn = SimpleRNN(units=RNN_HIDDEN_DIM,
                           dropout=DROPOUT_P,
                           recurrent_dropout=RNN_DROPOUT_P,
                           return_sequences=True,
                           return_state=True)
    model = ModelToInspect(vocab_size=vocab_size,
                           embedding_dim=EMBEDDING_DIM,
                           rnn_cell=simple_rnn,
                           hidden_dim=HIDDEN_DIM,
                           dropout_p=DROPOUT_P,
                           num_classes=len(classes))
    model.sample(x_in_shape=(sample_X.shape[1],), seq_lengths_shape=(2,))

    z = model(sample_inputs)
    print(f"z: {z} ==> shape: {z.shape}")

    model = TextClassificationRNNModel(vocab_size=vocab_size,
                                       embedding_dim=EMBEDDING_DIM,
                                       rnn_cell=simple_rnn,
                                       hidden_dim=HIDDEN_DIM,
                                       dropout_p=DROPOUT_P,
                                       num_classes=len(classes))
    model.sample(x_in_shape=(sequence_size,), seq_lengths_shape=(2,))

    # Compile
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_CRITERIA, verbose=1, mode='min'),
                 ReduceLROnPlateau(patience=1, factor=0.1, verbose=0),
                 TensorBoard(log_dir='tensorboard/simple_rnn', histogram_freq=1, update_freq='epoch')]

    # Training
    training_history = model.fit_generator(generator=training_generator,
                                           epochs=NUM_EPOCHS,
                                           validation_data=validation_generator,
                                           callbacks=callbacks,
                                           shuffle=False,
                                           class_weight=class_weights,
                                           verbose=1)
    # Evaluation
    testing_history = model.evaluate_generator(generator=testing_generator,
                                               verbose=1)
if __name__ == '__main__':
    main()
