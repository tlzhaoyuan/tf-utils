from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def untokenize(indices, tokenizer):
    """Untokenize a list of indices into string."""
    return " ".join([tokenizer.index_word[index] for index in indices])


def encode_label(y_train, y_val, y_test, verbose=False):
    y_encoder = LabelEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    y_val = y_encoder.transform(y_val)
    y_test = y_encoder.transform(y_test)
    classes = y_encoder.classes_
    counts = np.bincount(y_train)
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}
    if verbose:
        # Class weights
        print(f"class counts: {counts},\nclass weights: {class_weights}")
    return y_train, y_val, y_test, classes, class_weights

def load_glove_embeddings(embeddings_file):
    """Load embeddings from a file."""
    embeddings = {}
    with open(embeddings_file, "r") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings

def make_embeddings_matrix(embeddings, word_index, embedding_dim, verbose=False):
    """Create embeddings matrix to use in Embedding layer."""
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    if verbose:
        print(f"<Embeddings(words={embedding_matrix.shape[0]}, dim={embedding_matrix.shape[1]})>")
    return embedding_matrix

def tokenize(X_train, X_val, X_test, filters, lower, char_level, verbose=False, one_hot=False):
    tokenizer = Tokenizer(
        filters=filters,
        lower=lower,
        char_level=char_level,
        oov_token='<UNK>'
    )
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    original_text = X_train[0]
    X_train = np.array(tokenizer.texts_to_sequences(X_train))
    X_val = np.array(tokenizer.texts_to_sequences(X_val))
    X_test = np.array(tokenizer.texts_to_sequences(X_test))
    if verbose:
        preprocessed_text = untokenize(X_train[0], tokenizer)
        print(f"{original_text} \n\t→ {preprocessed_text} \n\t→ {X_train[0]}")
    if one_hot:
        X_train = np.array([to_categorical(seq, num_classes=vocab_size) for seq in X_train])
        X_val = np.array([to_categorical(seq, num_classes=vocab_size) for seq in X_val])
        X_test = np.array([to_categorical(seq, num_classes=vocab_size) for seq in X_test])
    return X_train, X_val, X_test, vocab_size

