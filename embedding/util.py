import matplotlib.pyplot as plt

from gensim.models import KeyedVectors

from sklearn.decomposition import PCA


def plot_embeddings(words, embeddings, pca_results):
    plt.title('word embedding')
    for word in words:
        index = embeddings.index2word.index(word)
        plt.scatter(pca_results[index, 0], pca_results[index, 1])
        plt.annotate(word, xy=(pca_results[index, 0], pca_results[index, 1]))
    plt.show()


def load_glove(embedding_dim):
    word2vec_file = '../data/glove.6B.{0}.word2vec'.format(embedding_dim)
    glv = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    return glv


if __name__ == '__main__':
    EMBEDDING_DIM = 100
    glove = load_glove(EMBEDDING_DIM)
    # Reduce dimensionality for plotting
    X = glove[glove.vocab]
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    # Visualize
    plot_embeddings(words=["king", "queen", "man", "woman"],
                    embeddings=glove,
                    pca_results=pca_results)
