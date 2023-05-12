import pickle
from datetime import date, datetime

from django.core.files.base import File
from django.conf import settings
from django.apps import apps
from ml.models import Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def load_loan_dataset(verbose=True):
    Loan = apps.get_model('loans', 'Loan')
    qs = Loan.objects.ml_data()
    qs = qs.values('state', 'term', 'no_emp','created_jobs','retained_jobs', 
                     'gross_appv', 'recession', 'secured_loan', 'gov_secured',
                     'is_rural', 'low_doc', 'new_business', 'econ_sector',
                     'inflation_on_loan','unemployment_on_loan', 'defaulted')
    return qs
    
def prepare_data(qs, verbose = True):
    df = pd.DataFrame(qs)
    lb = LabelBinarizer()
    objList = df.select_dtypes(include = "bool").columns
    for feature in objList:
        df[feature] = lb.fit_transform(df[feature])
    df['gov_secured'] = df['gov_secured'].astype(float)
    df['gross_appv'] = df['gross_appv'].astype(float)
    if verbose:
        print(f"{df.shape[0]} loans were loaded for training.")
    return df

def train_xgboost_model(cv=5, n_iter=50, scoring='recall', verbose=True):
    start_time = datetime.now() 
    qs = load_loan_dataset(verbose=verbose)
    df = prepare_data(qs=qs, verbose=True)
    X, y = df.drop(['defaulted'], axis= 1), df['defaulted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    bst = xgb.XGBClassifier() 
    search_space = {
        'max_depth': Integer(3,10),
        'min_child_weight': Integer(0, 20),
        'learning_rate': Real(0.05, 1), 
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'colsample_bylevel': Real(0.5, 1.0),
        'colsample_bynode' : Real(0.5, 1.0),
        'alpha': Real(0.0, 10.0),
        'reg_lambda': Real(0.0, 10.0),
        'gamma': Real(0.0, 5),
        }
    opt = BayesSearchCV(bst, search_space, cv=cv, n_iter=n_iter, scoring='recall')
    opt.fit(X_train, y_train)
    if verbose:
        print(f"Bayes Search was performed successfully")
    xgb_params = opt.best_params_
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    if verbose:
        print(f"Model was trained successfully")
    y_pred = model.predict(X_test)
    recall = recall_score(y_pred, y_test, pos_label=1)
    precision = precision_score(y_pred, y_test, pos_label=1)
    accuracy = accuracy_score(y_pred, y_test)
    acc_label = round((100* accuracy),2)
    today = str(date.today())
    model_name = f"model-{acc_label}-{today}" 
    binary_data = pickle.dumps(model)
    time_elapsed = datetime.now() - start_time
    obj = Model.objects.create(
        model_name = model_name,
        model = binary_data,
        accuracy = round(accuracy, 2),
        recall = round(recall, 2),
        precision = round(precision, 2),
        train_samples = len(X_train),
        test_samples = len(X_test),
        train_time = str(time_elapsed),
        max_depth = xgb_params['max_depth'],
        min_child_weight = xgb_params['min_child_weight'],
        learning_rate = xgb_params['learning_rate'],
        subsample = xgb_params['subsample'],
        colsample_bytree = xgb_params['colsample_bytree'],
        colsample_bylevel = xgb_params['colsample_bylevel'],
        colsample_bynode = xgb_params['colsample_bynode'],
        alpha = xgb_params['alpha'],
        reg_lambda = xgb_params['reg_lambda'],
        gamma = xgb_params['gamma'],
    )
    if verbose:
        print(f"New model: {model_name}, successfully created")

def load_model():
    raw_model = Model.objects.filter(active=True)
    if raw_model.exists():
        if len(raw_model) == 1:
            model = pickle.loads(raw_model[0].model)
        else:
            raw_model = raw_model.latest('created')
            model = pickle.loads(raw_model[0].model)        
    else:
        raw_model = Model.objects.latest('created')
        model = pickle.loads(raw_model[0].model) 

    return model

def optimize_loan_parameters(df:pd.DataFrame, bst: xgb.XGBClassifier):
    term = df['term'][0]
    gross_appv = df['gross_appv'][0]
    loan_class = bst.predict(df)[0]
    loan_class_prob = bst.predict_proba(df)[0][loan_class]
    # Adjust second term to expand the testing range
    gross_margin = gross_appv * 0.20
    gross_max_limit = gross_appv + gross_margin
    gross_min_limit = gross_appv - gross_margin
    # Adjust second term to expand the testing range
    term_margin = term * 0.3
    term_max_limit = term+term_margin
    term_min_limit = term-term_margin
    # Is term_min_limit is less than a year, 
    if term_min_limit/12 < 1:
        term_min_limit = term
    test_samples = np.linspace(gross_min_limit, gross_max_limit, 10).round(0).astype(int).tolist()
    # Get gross approved candidate values rounded to the nearest 1000
    test_samples = [int(round((x/1000),0)*1000) for x in test_samples]
    term_samples = np.linspace(term_min_limit, term_max_limit, 10).round(0).astype(int).tolist()
    # Get terms that are only multiples of 12 which mean they are complete years
    term_samples = list(set([int(round((x/12),0)*12) for x in term_samples]))
    combinations = [(x,y) for x in test_samples for y in term_samples]
    num_combinations = [*range(0,len(combinations),1)]
    samples = dict(zip(num_combinations, combinations))
    probabilities = []
    for key in samples:
        test_frame = df
        test_frame['gross_appv'] = samples[key][0] 
        test_frame['term'] = samples[key][1]
        probabilities.append(bst.predict_proba(test_frame)[0][loan_class])
    max_prob = max(probabilities)
    max_prob_index = probabilities.index(max_prob)
    if max_prob > loan_class_prob:
        prob_improv = max_prob - loan_class_prob 
        return (samples[max_prob_index], max_prob, loan_class_prob,prob_improv)
    else:
        return None   
