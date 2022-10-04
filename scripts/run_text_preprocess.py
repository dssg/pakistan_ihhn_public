import re
import nltk
import string
import logging
import pandas as pd
import multiprocessing as mp

from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from typing import List

from src.utils.utils import get_data


def count_question_marks(text):
    """counts number of questionmarks"""
    return len(text) - len(text.rstrip("?"))


def strip_html_tags(text):
    """removes HTML tags in the text"""
    soup = BeautifulSoup(text, "html.parser").text
    return soup


def strip_urls(text):
    """Strips any URLs in the text (has to be prepended by http/www)"""
    t = re.sub(r"(http|www)\S+", "", text)
    return t


def basic_denoising(text):
    """Runs some basic denoising steps on all texts"""

    text = text.lower()
    text = strip_html_tags(text)
    text = strip_urls(text)

    return text


def remove_numbers(text):
    """strip numbers from the text"""
    try:
        words = word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        words = word_tokenize(text)

    pattern = r"[0-9]"
    # words = ['' if w.isdigit() else w for w in words]
    words = [re.sub(pattern, "", w) for w in words]

    return " ".join(words)


def remove_punctuation(text, ignore_punct=[]):
    """replace punctuation with whitespace"""
    punct = string.punctuation.translate(str.maketrans("", "", "".join(ignore_punct)))
    translator = str.maketrans("", "", punct)
    try:
        stripped = [w.translate(translator) for w in word_tokenize(text)]
    except LookupError:
        nltk.download("punkt")
        stripped = [w.translate(translator) for w in word_tokenize(text)]

    return " ".join(stripped)


def stem_words(text):
    """combines the different forms of the words into one (verbs/adverbs/adjectives)"""
    # text = text.split()
    try:
        stemmer = LancasterStemmer()
    except LookupError:
        nltk.download("wordnet")

    try:
        words = word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        words = word_tokenize(text)

    stems = [stemmer.stem(w) for w in words]
    return " ".join(stems)


def lemmatize_words(text):
    """converts the word into its root form"""
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # TODO: not an elegant solution
    except LookupError:
        nltk.download("wordnet")
        lemmatizer = WordNetLemmatizer()

    try:
        words = word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        words = word_tokenize(text)

    lemmas = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(lemmas)


def remove_stop_words(text, custom_stop_words=None):
    """Remove the generic stop wods like articles, prepositions from text"""
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))

    # Removing stop words
    try:
        words = [w for w in word_tokenize(text) if w not in stop_words]
    except LookupError:
        nltk.download("punkt")
        words = [w for w in word_tokenize(text) if w not in stop_words]

    # Removing custom stopwords
    if custom_stop_words is not None:
        words = [w for w in word_tokenize(text) if w not in custom_stop_words]

    return " ".join(words)


def remove_custom_stopwords(text, stopword_file):
    """remove the custom stop words from text
    Stopwords should be in a text file separated by new file
    """
    with open(stopword_file) as f:
        content = f.read()
        stopwords = content.split("\n")

    words = [w for w in word_tokenize(text) if w not in stopwords]

    return " ".join(words)


def process_target(
    texts: pd.Series,
    processing_functions: List,
    preprocessed_list,
    stopword_file=None,
) -> pd.Series:
    """The target function for the process. Preprocesses the given subset of texts"""

    process_name = mp.current_process().name

    logging.info(
        "{}:Starting preprocessing {} documents".format(process_name, texts.shape[0])
    )

    for i, proc_func in enumerate(processing_functions):
        logging.info("{}: Performing {} on text".format(process_name, proc_func))

        if i == 0:
            return_series = texts.apply(proc_func)
        else:
            return_series = return_series.apply(proc_func)

    preprocessed_list.append(return_series)


# TODO: Handle removing custom stop words
def run_preprocessing_steps(
    texts: pd.Series, processing_steps: List, n_jobs=-1
) -> pd.Series:
    """Run the preprocessing steps for the raw text in the given
    order and return the preprocessed text

    Args:
        texts (pd.Series): The text to be preprocessed.
        steps (List[str]): The steps to be run. Should be steps included in
        the preprocessing dict
    """

    preprocess_dict = {
        "remove_numbers": remove_numbers,
        "remove_punctuation": remove_punctuation,
        "remove_stop_words": remove_stop_words,
        "stem": stem_words,
        "lemmatize": lemmatize_words,
    }

    # The mapping functions for each preprocessing step
    processing_functions = [preprocess_dict[x] for x in processing_steps]

    if (n_jobs > mp.cpu_count()) or n_jobs == -1:
        n_jobs = mp.cpu_count()

    num_docs = texts.shape[0]
    logging.info(
        "Starting multicore preprocessing of {} docs with {} processes".format(
            num_docs, n_jobs
        )
    )

    # Assigning jobs
    chunk_size, mod = divmod(num_docs, n_jobs)
    chunks_list = [chunk_size] * n_jobs

    # Distributing the remainder
    for i in range(mod):
        chunks_list[i] = chunks_list[i] + 1

    logging.info("Job assignments : {}".format(chunks_list))
    print("test")

    # Multiprocessing
    manager = mp.Manager()
    results_list = manager.list()

    idx_cursor = 0
    jobs = []

    for i, chunk_size in enumerate(chunks_list):
        p = mp.Process(
            name=f"p{i}",
            target=process_target,
            kwargs={
                "texts": texts.iloc[idx_cursor : (idx_cursor + chunk_size)],
                "processing_functions": processing_functions,
                "preprocessed_list": results_list,
            },
        )

        jobs.append(p)
        p.start()

        idx_cursor = idx_cursor + chunk_size

    for proc in jobs:
        proc.join()

    preprocessed_data = pd.concat(results_list, axis=0)
    preprocessed_data.rename("preprocessed", inplace=True)

    logging.info(preprocessed_data.head())

    logging.info(
        "preprocessed and returning {} docs".format(preprocessed_data.shape[0])
    )

    return preprocessed_data


# calculate for each value within ed_dx
dx = get_data("""select ed_dx from model_output.train;""")

question_mark_count = []
cleaned_text = []

run_preprocessing_steps(
    dx["ed_dx"],
    ["remove_numbers", "remove_punctuation", "remove_stop_words", "stem", "lemmatize"],
)
for i in range(len(dx)):
    question_mark_count.append(count_question_marks(dx.ed_dx[i]))
    clean_text = strip_html_tags(dx.ed_dx[i])
    clean_text = strip_urls(clean_text)
    clean_text = basic_denoising(clean_text)
    clean_text = remove_numbers(clean_text)
    clean_text = remove_punctuation(clean_text)
    clean_text = stem_words(clean_text)
    clean_text = lemmatize_words(clean_text)
    clean_text = remove_stop_words(clean_text, custom_stop_words=None)
    cleaned_text.append(clean_text)

dx["question_mark_count"] = question_mark_count
