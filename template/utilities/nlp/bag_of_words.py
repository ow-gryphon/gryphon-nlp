from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import text


class BagOfWords:

    @staticmethod
    def fit_regular_bow(data: pd.Series) -> Tuple[text.CountVectorizer, np.array]:

        # Bag of words
        bow = text.CountVectorizer()
        bow_result = bow.fit_transform(data.fillna(''))

        return bow, bow_result

    @staticmethod
    def fit_tfidf_bow(data: pd.Series):

        # Bag of words
        bow = text.CountVectorizer()
        bow_result = bow.fit_transform(data.fillna(''))

        # TFIDF
        tfidf = text.TfidfTransformer(norm=None)
        tfidf_result = tfidf.fit_transform(bow_result)

        return tfidf, tfidf_result

    @staticmethod
    def fit_normalized_bow(data: pd.Series):

        # Bag of words
        bow = text.CountVectorizer()
        bow_result = bow.fit_transform(data.fillna(''))

        # Normalization
        normalization_result = preprocessing.normalize(bow_result, axis=0)

        return normalization_result
