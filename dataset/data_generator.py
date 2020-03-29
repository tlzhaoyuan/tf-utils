import math

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class DataGenerator(Sequence):
    """Custom data loader."""

    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """# of batches."""
        return math.ceil(len(self.X) / self.batch_size)

    def __str__(self):
        return (f"<DataGenerator("
                f"batch_size={self.batch_size}, "
                f"batches={len(self)}, "
                f"shuffle={self.shuffle})>")

    def __getitem__(self, index):
        """Generate a batch."""
        # Gather indices for this batch
        batch_indices = self.epoch_indices[
                        index * self.batch_size:(index + 1) * self.batch_size]

        # Generate batch data
        inputs, outputs = self.create_batch(batch_indices=batch_indices)

        return inputs, outputs

    def on_epoch_end(self):
        """Create indices after each epoch."""
        self.epoch_indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.epoch_indices)

    def create_batch(self, batch_indices):
        """Generate batch from indices."""
        # Get batch data
        X = self.X[batch_indices]
        y = self.y[batch_indices]

        seq_lengths = np.array([[i, len(x) - 1] for i, x in enumerate(X)])

        # Pad batch
        max_seq_len = max([len(x) for x in X])
        X = pad_sequences(X, padding="post", maxlen=max_seq_len)

        return [X, seq_lengths], y


class DataGeneratorV2(Sequence):
    """Custom data loader."""

    def __init__(self, X, y, batch_size, vocab_size, max_filter_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_filter_size = max_filter_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """# of batches."""
        return math.ceil(len(self.X) / self.batch_size)

    def __str__(self):
        return (f"<DataGenerator("
                f"batch_size={self.batch_size}, "
                f"batches={len(self)}, "
                f"shuffle={self.shuffle})>")

    def __getitem__(self, index):
        """Generate a batch."""
        # Gather indices for this batch
        batch_indices = self.epoch_indices[
                        index * self.batch_size:(index + 1) * self.batch_size]

        # Generate batch data
        X, y = self.create_batch(batch_indices=batch_indices)

        return X, y

    def on_epoch_end(self):
        """Create indices after each epoch."""
        self.epoch_indices = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.epoch_indices)

    def create_batch(self, batch_indices):
        """Generate batch from indices."""
        # Get batch data
        X = self.X[batch_indices]
        y = self.y[batch_indices]

        # Pad batch
        max_seq_len = max(self.max_filter_size, max([len(x) for x in X]))
        X = pad_sequences(X, padding="post", maxlen=max_seq_len)

        return X, y
