import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Bunch(dict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


def load_json(rpath):
    ''' Load json file. '''
    try:
        with open(rpath, "r") as f:
            json_report = json.load(f)
        return json_report
    except ValueError:
        print "Failed to load json file: " + rpath
        return None


def get_calls(json_report):
    ''' get all API calls from report (include all processes) '''
    calls = []
    try:
        processes = json_report['behavior']['processes']
        for process in processes:
            if process["process_name"].__str__() == "lsass.exe":    # non-informative process
                continue
            for call in process["calls"]:
                term = call['api'].__str__().lower()
                if term == '__exception__':
                    continue
                calls.append(term)
        return ' '.join(calls)
    except (KeyError, AttributeError, IndexError, UnicodeEncodeError), e:
        print e.message


def get_raw_data(path, mode):

    if mode == 'train':
        data = {'X': [], 'Y': []}
        for family in os.listdir(path):
            if os.path.isdir(os.path.join(path, family)):
                for report in os.listdir(os.path.join(path, family)):
                    json_report = load_json(os.path.join(path, family, report))
                    data['X'].append(get_calls(json_report))
                    data['Y'].append(family)
        return data

    elif mode == 'predict':
        data = []
        if os.path.isfile(path):
            json_report = load_json(os.path.join(path))
            data.append(get_calls(json_report))
        return data


def get_feature_vector(data, n, max_features):
    ''' create Feature Vector using scikit TF-IDF text analyzer.
     this function create feature vector of length - max_features '''
    tf = TfidfVectorizer(analyzer='word', ngram_range=(n, n), smooth_idf=True,
                         max_features=int(max_features), norm='l2', use_idf=True)
    tf.fit_transform(data)
    return tf.vocabulary_


def wrapper(func, args, res):
    ''' Wrapper to get return variable from thread '''
    res.append(func(*args))


def get(path, mode='train', n=2, max_features=100, fv=None):
    data = {'X': [], 'Y': []}
    counter = 0
    lock = threading.Lock()
    threads = []
    try:
        if mode == 'train':
            for family in os.listdir(path):
                if os.path.isdir(os.path.join(path, family)):
                    for report in os.listdir(os.path.join(path, family)):
                        json_report = load_json(os.path.join(path, family, report))     # load report
                        calls = []
                        t = threading.Thread(target=wrapper, args=(get_calls, (json_report,), calls))
                        threads.append(t)
                        t.start()
                        t.join()
                        if not calls[0]:
                            continue
                        with lock:
                            data['X'].append(calls[0])
                            data['Y'].append(family)
                            counter += 1
                        if not counter % 50:
                            print str(counter)
            fv = get_feature_vector(data['X'], n, max_features)
            tf = TfidfVectorizer(analyzer='word', ngram_range=(n, n), smooth_idf=True,
                                 vocabulary=fv, norm='l2', use_idf=False)
            tf_matrix = tf.fit_transform(data['X'])

        elif mode == 'predict':
            if os.path.isdir(path):
                for report in os.listdir(path):
                    if os.path.isfile(os.path.join(path, report)):
                        json_report = load_json(os.path.join(path, report))
                        calls = get_calls(json_report)
                        if not calls:
                            continue
                        data['X'].append(calls)
                        data['Y'].append(None)
            else:
                json_report = load_json(os.path.join(path))
                calls = get_calls(json_report)
                if calls:
                    data['X'].append(calls)
                    data['Y'].append(None)

            tf = TfidfVectorizer(analyzer='word', ngram_range=(n, n), smooth_idf=False,
                                 vocabulary=fv, norm='l2', use_idf=False)
            tf_matrix = tf.fit_transform(data['X'])

        # Data information #
        # Dimensions
        n_samples = int(tf_matrix.shape[0])
        n_features = int(tf_matrix.shape[1])
        # Features
        feature_names = np.array(list(tf.get_feature_names()))
        features = np.empty((n_samples, n_features))
        feature_vector = fv
        # Classes
        target_names = np.array(list(set(data['Y'])))
        target = np.empty((n_samples,), dtype=np.int)

        i = 0
        for row in tf_matrix:
            features[i] = row.A[0]
            target[i] = np.asarray(list(target_names).index(data['Y'][i]), dtype=np.int)
            i += 1

        return Bunch(data=features,
                     target=target,
                     target_names=target_names,
                     feature_names=feature_names,
                     feature_vector=feature_vector)
    except:
        logger.error(" Somthing went wrong.")
        exit(1)
