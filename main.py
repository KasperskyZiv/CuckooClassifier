import os
import json
from optparse import OptionParser
from sklearn.ensemble import RandomForestClassifier
import extractFeatures
import trainModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.getcwd(), 'Models')


def logo():
    print "-----------------------"
    print "CuckooClassifier - 2017"
    print "-----------------------\n"


def opt_prase(args=None):
    parser = OptionParser()
    parser.add_option("-p",
                      "--predict",
                      action="store_false",
                      dest="train_model",
                      default=False)
    parser.add_option("-t",
                      "--train",
                      action="store_true",
                      dest="train_model",
                      default=False)
    parser.add_option('-v', '--verbose',
                      dest="verbose",
                      default=False,
                      action="store_true",
                      help="print status messages to screen"
                      )
    parser.add_option("-r",
                      "--reports",
                      type="string",
                      action="store",
                      dest="path",
                      help="path to report/folder for train or predict"
                      )
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def list_models():
    '''list prev saved models for selection'''
    all_models = []
    try:
        with open(os.path.join(MODELS_DIR, 'models_db.json'), 'r') as f:
            f_handler = f.read().splitlines()
        i = 1
        for model in f_handler:
            if model:
                all_models.append(model)
                model = json.loads(model)
                print str(i) + ") " + model['name'] + ', ' + model['time'] + ', ' + model['desc']
                i += 1
    except IOError:
        logging.error(" No models founds, please train a model.")
    except ValueError:
        pass
    finally:
        return all_models


def main():
    (options, args) = opt_prase()
    logo()
    if options.train_model:     # Train
        dataset = extractFeatures.get(options.path, 'train', 2, 100)
        model = trainModel.Model()
        model.create_model(RandomForestClassifier(n_estimators=200, random_state=10, n_jobs=-1))
        model.fit_model(dataset)
        model.export_model(MODELS_DIR)
        model.cross_validation()
    else:                       # Predict
        print "Loading saved models..."
        print "-----------------------"
        models = list_models()
        idx = raw_input("\nSelect Model #: ")
        model = trainModel.Model()
        model.import_model(json.loads(models[int(idx)-1]))
        dataset = extractFeatures.get(options.path, 'predict', 2, 100, model.fv)
        logger.info("Prediction - " + model.predict_class(dataset, json.loads(models[int(idx)-1])))

if __name__ == "__main__":
    main()
