from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import json
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


class Model:
    def __init__(self):
        self.name = 'None'
        self.desc = 'None'
        self.dataset = None
        self.model = None
        self.cross_val_score = None
        self.train_test_score = None
        self.fv = None

    def create_model(self, model):
        ''' Declare a new model. '''
        self.model = model
        self.name = self.model.__str__().split('(')[0] + datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info("Creating new model: "
                    + self.name
                    + '.pkl\n')

    def fit_model(self, dataset):
        ''' Fit model with data for the selected algorithm '''
        self.dataset = dataset
        self.model.fit(self.dataset.data, self.dataset.target)

    def cross_validation(self, k=5):
        ''' Stratified cross-validation - split the data such that the proportions between classes
        are the same in each fold as they are in the whole dataset.'''
        skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        scores = cross_val_score(self.model, self.dataset.data, self.dataset.target, cv=skfold)
        self.cross_val_score = scores.mean()
        print self.cross_val_score

    def train_test(self, size=0.33):
        ''' train test split of 33% '''
        x_train, x_test, y_train, y_test = train_test_split(self.dataset.data,
                                                            self.dataset.target,
                                                            test_size=size,
                                                            random_state=0)
        y_pred = self.model.fit(x_train, y_train).predict(x_test)
        self.train_test_score = accuracy_score(y_test, y_pred)

    def export_model(self, model_path=None):
        ''' Export model as pkl file with metadata saved to
        models_db.json @ Models dir '''
        self.desc = ', '.join([self.model.__str__().split('(')[0],
                              str(len(self.dataset.target_names)) + " Classes",
                              str(self.dataset.data.shape[0]) + " Instances",
                              str(self.dataset.data.shape[1]) + " Features"])
        new_model = {
            'path': os.path.join(model_path, self.name + '.pkl'),
            'time': datetime.now().strftime('%Y_%m_%d %H:%M:%S'),
            'name': self.name,
            'desc': self.desc,
            'class_names': ', '.join(self.dataset.target_names),
            'class_count': len(self.dataset.target_names),
            'instances': self.dataset.data.shape[0],
            'features': self.dataset.data.shape[1],
            'feature_vector': self.dataset.feature_vector
        }
        joblib.dump(self.model, new_model['path'])
        # Writing model JSON data
        if not os.path.isfile(os.path.join(model_path, 'models_db.json')):
            open(os.path.join(model_path, 'models_db.json'), 'a').close()
        with open(os.path.join(model_path, 'models_db.json'), 'r+') as f:
            content = f.read() + '\n'
            f.seek(0, 0)
            f.write(json.dumps(new_model).rstrip('\r\n') + '\n' + content)

        logger.info("Model created:\n"
                    + str(self.model.__str__().split('(')[0]) + "\n"
                    + str(len(self.dataset.data)) + " instances.\n"
                    + str(len(self.dataset.feature_names)) + " features.\n"
                    + str(len(self.dataset.target_names)) + " classes.\n")

    def import_model(self, model=None):
        '''Import model for prediction'''
        self.name = model['name']
        self.desc = model['desc']
        self.fv = model['feature_vector']
        self.model = joblib.load(str(model['path']))

    def predict_class(self, X, model):
        '''Predict the class for given instance using the current model'''
        predict_proba = self.model.predict_proba(X.data)
        return model['class_names'].split(',')[predict_proba[0].argmax()] + ': ' + str(predict_proba[0].max()*100) + "%"
