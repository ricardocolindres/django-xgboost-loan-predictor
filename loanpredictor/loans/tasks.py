from dateutil.relativedelta import relativedelta
from datetime import date
from .models import Loan
from celery import shared_task

@shared_task(name = 'update_active_loans')
def update_active_loans(verbose=True):
    queryset = Loan.objects.filter(active=True)
    now = date.today()
    TIME_AFTER_MATURITY_ACTIVE = 12
    updated = 0
    for obj in queryset:
        if obj.approval_date:
            active_frame = obj.maturity_date + relativedelta(months=+TIME_AFTER_MATURITY_ACTIVE)
            if active_frame < now or obj.defaulted:
                obj.active = False
                updated += 1
    Loan.objects.bulk_update(queryset, ['active'])
    if verbose:
        print(f"Updated {updated} loans")