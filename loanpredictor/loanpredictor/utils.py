import os
import csv
from . import settings
from django.utils import timezone
import datetime
import pytz

TIME_AFTER_MATURITY_ACTIVE = 365
LOAN_DATA = os.path.join(settings.DATA_DIR, 'cold_load.csv')

def load_loan_data(limit=0, verbose=False):
    with open(LOAN_DATA) as csvfile:
        reader = csv.DictReader(csvfile)
        dataset = []
        for i, row in enumerate(reader):
            utc = pytz.UTC
            now = timezone.now()
            active_frame = utc.localize(datetime.datetime.strptime(row.get("loan_maturity_date"), '%Y-%m-%d') + datetime.timedelta(days=TIME_AFTER_MATURITY_ACTIVE))
            if active_frame < now:
                active = False
            else:
                active = True
            data = {'internal_id' : row.get("loan_id"),
                    'bs_name' : row.get("name"),
                    'approval_date' : row.get("approval_date"),
                    'maturity_date' : row.get("loan_maturity_date"),
                    'state' : row.get("state"),
                    'term' : row.get("term"), 
                    'no_emp' : row.get("no_emp"),
                    'created_jobs' : row.get("created_jobs"),
                    'retained_jobs': row.get("retained_jobs"),
                    'gross_appv' : row.get("gross_appv"),
                    'recession' : row.get("recession"),
                    'secured_loan' : row.get("secured_loan"),
                    'gov_secured' : row.get("%_gov_secured"),
                    'defaulted' : row.get("defaulted"),
                    'is_rural' : row.get("is_rural"), 
                    'low_doc' : row.get("low_doc"),
                    'new_business' : row.get("new_business"),
                    'econ_sector' : row.get("econ_sector"),
                    'inflation_on_loan' : row.get("inflation_on_loan"), 
                    'unemployment_on_loan' : row.get("unemployment_on_loan"),
                    'active' : active,
                    'approved':True
                }
            dataset.append(data)
            if limit == 0:
                continue
            else:
                if i + 1 == limit:
                    if verbose:
                        print(f'Selected limit ({limit}) reached')
                    break
        if verbose:
            print(f'{len(dataset)} datapoint(s) were successfully read')

        return dataset