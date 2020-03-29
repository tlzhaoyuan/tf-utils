from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Concatenate, Dense, Dropout, BatchNormalization, \
    Activation
from tensorflow.keras import Input
import tensorflow as tf

class TextClassificationCNNModel(Model):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters,
                 hidden_dim, dropout_p, num_classes, freeze_embeddings=False):
        super().__init__()

        # Embeddings
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   trainable=not freeze_embeddings)
        # Convolutional filters
        self.convs = []
        self.pools = []
        for filter_size in filter_sizes:
            conv = Conv1D(filters=num_filters, kernel_size=filter_size,
                          padding='same', activation='relu')
            pool = GlobalMaxPool1D(data_format='channels_last')
            self.convs.append(conv)
            self.pools.append(pool)

        # Concatenation
        self.concat = Concatenate(axis=1)

        # FC layers
        self.fc1 = Dense(units=hidden_dim, activation='relu')
        self.dropout = Dropout(rate=dropout_p)
        self.fc2 = Dense(units=num_classes, activation='softmax')

    def call(self, x_in, training=False, **kwargs):
        """Forward pass.
        :param **kwargs:
        """

        # Embed
        x_emb = self.embedding(x_in)

        # Convolutions
        convs = []
        for i in range(len(self.convs)):
            z = self.convs[i](x_emb)
            z = self.pools[i](z)
            convs.append(z)

        # Concatenate
        z_cat = self.concat(convs)

        # FC
        z = self.fc1(z_cat)
        if training:
            z = self.dropout(z, training=training)
        y_pred = self.fc2(z)

        return y_pred

    def sample(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=x, outputs=self.call(x, )).summary()


class TextClassificationCNNModelV2(Model):
    def __init__(self, filter_size, num_filters,
                 hidden_dim, dropout_p, num_classes):
        super().__init__()

        # Convolutional filters
        self.conv = Conv1D(filters=num_filters, kernel_size=filter_size, padding='same')
        self.relu = Activation('relu')
        self.batch_norm = BatchNormalization()
        self.pool = GlobalMaxPool1D(data_format='channels_last')

        # FC layers
        self.fc1 = Dense(units=hidden_dim, activation='relu')
        self.dropout = Dropout(rate=dropout_p)
        self.fc2 = Dense(units=num_classes, activation='softmax')

    def call(self, x_in, training=False, **kwargs):
        """Forward pass.
            :param **kwargs:
            """

        # Cast input to float
        x_in = tf.cast(x_in, tf.float32)

        # Convolutions
        z = self.conv(x_in)
        z = self.relu(z)
        z = self.batch_norm(z)
        z = self.pool(z)

        # FC
        z = self.fc1(z)
        if training:
            z = self.dropout(z, training=training)
        y_pred = self.fc2(z)

        return y_pred

    def sample(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=x, outputs=self.call(x, )).summary()
