import tensorflow as tf
from . import MLP

if __name__ == '__main__':
    from util import MNISTLoader
    from tqdm import tqdm
    import numpy as np

    num_epochs = 5
    batch_size = 32
    learning_rate = 1e-3

    data_loader = MNISTLoader()
    model = MLP()
    optimizer = tf.keras.optimizers.Adam()

    num_iters = data_loader.train_data.shape[0] * num_epochs // batch_size
    for i in tqdm(range(num_iters)):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        print(f'index {i}, loss {loss.tonumpy()}')
    y_pred_raw = model.predict(data_loader.test_data)
    y_pred = y_pred_raw.argmax(axis=1)
    accuracy = np.mean(np.equal(y_pred, data_loader.test_label))
    print(f'accuracy is {accuracy}')
