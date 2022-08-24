import re
from typing import List

import pandas as pd

SPECIAL_CHARS = [
    ".", ",", ":", "\"", "=", "&", ";", "%", "$",
    "@", "%", "^", "*", "(", ")", "{", "}",
    "[", "]", "/", "\\", ">", "<", "-",
    "!", "?", ".", "'", '+', '–', '£', '€', '•',
    "--", "---", "#", "”", "“", "’", '‘', 'Î¸',
    "<br>", "\S*@\S*\s?", r'^https?:\/\/.*[\r\n]*',
    "xxxxxxxxxxxxx", "xxxxxxxxxxxx", "xxxxxxxxxxx",
    "xxxxxxxxxx", "xxxxxxxxx", "xxxxxxxx", "xxxxxxx",
    "xxxxxx", "xxxxx", "xxxx", "xxx", "xx"
]

CONTRACTIONS = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))

class TextPreparation:

    @staticmethod
    def lowercase(data: pd.Series) -> pd.Series:
        """
        Lower case the sentences.

        :param data: Pandas Series that will be transformed.

        :return: Resulting Pandas Series.
        :rtype: pd.Series
        """
        return data.apply(lambda x: str(x).lower())

    @staticmethod
    def remove_special_chars(data: pd.Series) -> pd.Series:
        """
        Removes all special characters from the sentences.

        :param data: Pandas Series that will be transformed.

        :return: Resulting Pandas Series.
        :rtype: pd.Series:
        """
        for remove in map(lambda r: re.compile(re.escape(r)), SPECIAL_CHARS):
            data.replace(remove, "", inplace=True)

        return data

    @staticmethod
    def remove_numbers(data: pd.Series) -> pd.Series:
        """
        Remove all numbers from the sentences.

        :param data: Pandas Series that will be transformed.

        :return: Resulting Pandas Series.
        :rtype: pd.Series:
        """
        return (
            data
            .apply(lambda x: re.sub(r'[0-9]', '', str(x)))
            .apply(lambda x: ' '.join(s for s in str(x).split() if not any(c.isdigit() for c in s)))
        )

    @staticmethod
    def remove_stopwords(data: pd.Series, language="english") -> pd.Series:
        """
        Remove stop words from the sentences.

        :param data: Pandas Series that will be transformed.
        :param language: Text idiom to remove stop words from.

        :return: Resulting Pandas Series.
        :rtype: pd.Series:
        """

        import nltk

        try:
            stopwords = nltk.corpus.stopwords.words(language)

        except LookupError:
            print("Downloading stop words from nltk package.")

            nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(language)

        # stopwords = list(set(stopwords) - set(['not']))
        for remove in stopwords:
            data.replace(re.compile(r'\b{}\b'.format(remove)), "", inplace=True)

        return data

    @staticmethod
    def get_proper_nouns(data: pd.Series) -> List[str]:
        """
        Get proper nouns from text Series

        :param data: Pandas Series that will be transformed.

        :return: Resulting list of proper nouns.
        :rtype: List[str]:
        """
        import nltk

        all_text = " ".join(data)
        all_words = list(set(all_text.split()))  # unique words

        try:
            tagged_sent = nltk.tag.pos_tag(all_words)
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            tagged_sent = nltk.tag.pos_tag(all_words)

        return list(map(lambda x: x[0], filter(lambda x: x[1] == 'NNP', tagged_sent)))

    @staticmethod
    def stem_text(data: pd.Series) -> pd.Series:
        """
        Stem and combine into string the sentences.

        :param data: Pandas Series that will be transformed.

        :return: Resulting Pandas Series.
        :rtype: pd.Series:
        """
        import nltk

        tokenizer = nltk.word_tokenize
        stemmer = nltk.PorterStemmer()

        try:
            return data.apply(
                lambda x: " ".join(map(lambda s: stemmer.stem(s), tokenizer(x)))
            )
        except LookupError:
            nltk.download('punkt')

    @staticmethod
    def remove_duplicated_spaces(data: pd.Series) -> pd.Series:
        """
        Stem and combine into string the sentences.

        :param data: Pandas Series that will be transformed.

        :return: Resulting Pandas Series.
        :rtype: pd.Series:
        """
        return data.apply(lambda x: " ".join(str(x).split()))

    @staticmethod
    def _expand_contractions(string: str, contractions=CONTRACTIONS):
        def replace(match):
            return contractions[match.group(0)]

        return contractions_re.sub(replace, str(string).lower())

    @classmethod
    def expand_contractions(cls, data: pd.Series):
        return data.apply(cls._expand_contractions)
