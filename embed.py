from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from dataset import load_agnews
from dataset.data_generator import DataGenerator
from embedding import untokenize, load_glove_embeddings, make_embeddings_matrix, encode_label
from model.text_classification_cnn import TextClassificationCNNModel
from util import train_val_test_split

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
SEED = 1234
SHUFFLE = True
DATA_FILE = '../data/news.csv'
INPUT_FEATURE = 'title'
OUTPUT_FEATURE = 'category'
FILTERS = "!\"'#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
LOWER = True
CHAR_LEVEL = False
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_EPOCHS = 10
BATCH_SIZE = 64
EMBEDDING_DIM = 100
NUM_FILTERS = 50
FILTER_SIZES = [2, 3, 4]
HIDDEN_DIM = 100
DROPOUT_P = 0.1
LEARNING_RATE = 1e-3
EARLY_STOPPING_CRITERIA = 3

def train_frozen(classes):
    FREEZE_EMBEDDINGS = True
    # Initialize model
    glove_frozen_model = TextClassificationCNNModel(vocab_size=vocab_size,
                                                    embedding_dim=EMBEDDING_DIM,
                                                    filter_sizes=FILTER_SIZES,
                                                    num_filters=NUM_FILTERS,
                                                    hidden_dim=HIDDEN_DIM,
                                                    dropout_p=DROPOUT_P,
                                                    num_classes=len(classes),
                                                    freeze_embeddings=FREEZE_EMBEDDINGS)
    glove_frozen_model.sample(input_shape=(10,))

if __name__ == '__main__':
    news = load_agnews(DATA_FILE)
    X = news[INPUT_FEATURE].values
    y = news[OUTPUT_FEATURE].values
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=X,
        y=y,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        shuffle=SHUFFLE
    )
    X_tokenizer = Tokenizer(
        filters=FILTERS,
        lower=LOWER,
        char_level=CHAR_LEVEL,
        oov_token='<UNK>'
    )

    X_tokenizer.fit_on_texts(X_train)

    # Convert text to sequence of tokens
    original_text = X_train[0]
    X_train = np.array(X_tokenizer.texts_to_sequences(X_train))
    X_val = np.array(X_tokenizer.texts_to_sequences(X_val))
    X_test = np.array(X_tokenizer.texts_to_sequences(X_test))
    preprocessed_text = untokenize(X_train[0], X_tokenizer)
    print(f"{original_text} \n\t→ {preprocessed_text} \n\t→ {X_train[0]}")

    # Dataset generator
    training_generator = DataGenerator(X=X_train,
                                       y=y_train,
                                       batch_size=BATCH_SIZE,
                                       max_filter_size=max(FILTER_SIZES),
                                       shuffle=SHUFFLE)
    validation_generator = DataGenerator(X=X_val,
                                         y=y_val,
                                         batch_size=BATCH_SIZE,
                                         max_filter_size=max(FILTER_SIZES),
                                         shuffle=False)
    testing_generator = DataGenerator(X=X_test,
                                      y=y_test,
                                      batch_size=BATCH_SIZE,
                                      max_filter_size=max(FILTER_SIZES),
                                      shuffle=False)
    print(f"training_generator: {training_generator}")
    print(f"validation_generator: {validation_generator}")
    print(f"testing_generator: {testing_generator}")

    y_train, y_val, y_test, classes = encode_label(y_train, y_val, y_test)
    embeddings_file = 'data/glove.6B.100d.txt.word2vec'
    glove_embeddings = load_glove_embeddings(embeddings_file=embeddings_file)
    embedding_matrix = make_embeddings_matrix(embeddings=glove_embeddings,
                                              word_index=X_tokenizer.word_index,
                                              embedding_dim=EMBEDDING_DIM)
    train_frozen(classes)