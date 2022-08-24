import gensim.downloader as api
import pandas as pd
from numpy import mean, stack, array


class WordEmbedding:

    def __init__(self, embedding_model='word2vec-google-news-300'):

        print("Loading embedding model to memory...")
        self.model = api.load(name=embedding_model)
        print("Done!")

    def get_sentence_vector(self, tokenized_text: str):
        """


        :param tokenized_text: String text to be transformed into a word embedding vector
        :return:
        """

        if type(tokenized_text) != str:
            tokenized_text = str(tokenized_text)

        vec = []
        for word in tokenized_text.split(' '):
            if word in self.model:
                vec.append(self.model[word])

        if len(vec) == 0:
            print(f"It wasn't possible to vectorize the sentence: \"{tokenized_text}\"")
            return 0 * self.model['cat']

        else:
            return mean(vec, axis=0)

    def transform(self, data: pd.Series) -> array:
        """
        Transform a given data series

        :param data: data Series with (n_samples) length
        :type: pd.Series

        :return: Array of shape (n_samples, n_features)
        :rtype: array
        """
        result = data.apply(self.get_sentence_vector)
        return stack(result.values, axis=0)
