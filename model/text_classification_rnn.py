import tensorflow as tf
from tensorflow.keras.layers import Embedding, Masking, Dense, Dropout, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as K


class TextClassificationRNNModel(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_dim, dropout_p, num_classes):
        super(TextClassificationRNNModel, self).__init__()

        # Embeddings
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   mask_zero=True,
                                   trainable=True)

        # Masking
        self.mask = Masking(mask_value=0.)

        # RNN
        self.rnn = rnn_cell

        # FC layers
        self.fc1 = Dense(units=hidden_dim, activation='relu')
        self.dropout = Dropout(rate=dropout_p)
        self.fc2 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs, training=False, **kwargs):
        """Forward pass.
        :param **kwargs:
        """

        # Inputs
        x_in, seq_lengths = inputs

        # Embed
        x_emb = self.embedding(x_in)

        # Masking
        z = self.mask(x_emb)

        # RNN
        z, hidden_state = self.rnn(x_emb)

        # Gather last relevant index
        z = tf.gather_nd(z, K.cast(seq_lengths, 'int32'))

        # FC
        z = self.fc1(z)
        if training:
            z = self.dropout(z, training=training)
        y_pred = self.fc2(z)

        return y_pred

    def sample(self, x_in_shape, seq_lengths_shape):
        x_in = Input(shape=x_in_shape)
        seq_lengths = Input(shape=seq_lengths_shape)
        inputs = [x_in, seq_lengths]
        return Model(inputs=inputs, outputs=self.call(inputs, )).summary()