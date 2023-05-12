from ml.utlis import train_xgboost_model
from celery import shared_task

@shared_task(name = 'train_model')
def train_model(cv=5, n_iter=1, scoring='recall', verbose=True):
    train_xgboost_model(cv=cv, n_iter=n_iter, scoring=scoring, verbose=verbose)