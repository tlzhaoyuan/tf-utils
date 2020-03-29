import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Input, Masking, Embedding
from tensorflow.keras import Model


class MLP(Model):
    def __init__(self, hidden_size, dropout_p, num_classes):
        super().__init__()
        self.fc1 = Dense(hidden_size, activation='relu')
        self.dropout = Dropout(rate=dropout_p)
        self.fc2 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        if training:
            x = self.dropout(x)
        outputs = self.fc2(x)
        return outputs

    def sample(self, input_shape):
        x = Input(shape=input_shape)
        return Model(inputs=x, outputs=self.call(x)).summary()


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(
            kernel_size=[5, 5],
            filters=32,
            padding='same',
            activation='relu'
        )
        self.pool1 = MaxPool2D(
            pool_size=[5, 5],
            strides=2
        )
        self.conv2 = Conv2D(
            kernel_size=[5, 5],
            filters=64,
            padding='same',
            activation='relu'
        )


class ModelToInspect(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_dim, dropout_p, num_classes):
        super(ModelToInspect, self).__init__()

        # Embeddings
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   mask_zero=True,
                                   trainable=True)

        # Masking
        self.mask = Masking(mask_value=0.)

        # RNN
        self.rnn = rnn_cell

    def call(self, inputs, training=False, **kwargs):
        """Forward pass.
        :param **kwargs:
        """

        # Forward pass
        x_in, seq_lengths = inputs
        x_emb = self.embedding(x_in)
        z = self.mask(x_emb)
        z, hidden_state = self.rnn(x_emb)

        return z

    def sample(self, x_in_shape, seq_lengths_shape):
        x_in = Input(shape=x_in_shape)
        seq_lengths = Input(shape=seq_lengths_shape)
        inputs = [x_in, seq_lengths]
        return Model(inputs=inputs, outputs=self.call(inputs, )).summary()