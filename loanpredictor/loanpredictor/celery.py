import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'loanpredictor.settings')

app = Celery('loanpredictor')
app.config_from_object("django.conf:settings", namespace='CELERY')

app.autodiscover_tasks()

app.conf.beat_schedule = {
    'train_model_24_hours': {
        'task' : 'train_model',
        'schedule' : 60 * 30, #30 Mins
        'kwargs' : {'cv':5, 
                    'n_iter':1, 
                    'scoring':'recall', 
                    'verbose':True}
    },
    'update_loan_active_24hours': {
        'task' : 'update_active_loans',
        'schedule' : 60 * 45, #45 Mins
        'kwargs' : {'verbose':True}
    }
}