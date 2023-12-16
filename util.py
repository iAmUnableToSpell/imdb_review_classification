import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import TextVectorization
import re
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean

unwanted = nltk.corpus.stopwords.words("english")

def prepare_data_dir():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    tf.keras.utils.get_file("aclImdb_v1", url,
        untar=True, cache_dir='.',
        cache_subdir='')

    # removes unsupervised (unlabeled) dir because we won't be using it.
    # This will throw an error on windows due to concurrent access and OS perms.
    # If you want to remove the unsupervised directory, delete it manually. It is not referenced in this project.

    # dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    # train_dir = os.path.join(dataset_dir, 'train')

    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)


# Custom standardization function for text vectorization layer
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# Evaluates a keras model
def evaluate_model_keras(model, test_ds, history):
    loss, accuracy, auc = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("AUC: ", auc)

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']



    epochs_to_graph = range(1, len(acc) + 1)

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].plot(epochs_to_graph, loss, 'r', label='Training loss')
    axs[0].plot(epochs_to_graph, val_loss, 'bo', label='Validation loss')
    axs[0].set_title('Epochs vs Loss', fontstyle='italic')
    axs[0].legend()

    axs[1].plot(epochs_to_graph, acc, 'r', label='Training accuracy')
    axs[1].plot(epochs_to_graph, val_acc, 'bo', label='Validation accuracy')
    axs[1].set_title('Epochs vs Accuracy', fontstyle='italic')
    axs[1].legend()

    plt.show()

def evaluate_nltk_preds(y, y_hat):
    acc = (y_hat == y).sum() / len(y)
    print("Accuracy: ", acc)

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True


def prepare_nltk(vocab_size):
    unwanted.extend([w.lower() for w in nltk.corpus.names.words()])
    positive_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
    )]
    negative_words = [word for word, tag in filter(
        skip_unwanted,
        nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
    )]
    # return positive_words, negative_words

    positive_fd = nltk.FreqDist(positive_words)
    negative_fd = nltk.FreqDist(negative_words)

    common_set = set(positive_fd).intersection(negative_fd)

    for word in common_set:
        del positive_fd[word]
        del negative_fd[word]

    top_positive = {word for word, count in positive_fd.most_common(vocab_size)}
    top_negative = {word for word, count in negative_fd.most_common(vocab_size)}
    return top_positive, top_negative


def nltk_extract_features(text, top_positive):
    sia = SentimentIntensityAnalyzer()
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features




