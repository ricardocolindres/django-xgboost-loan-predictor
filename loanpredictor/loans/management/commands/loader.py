from django.core.management.base import BaseCommand
from loanpredictor import utils
from loans.models import Loan


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("limit", nargs='?', default=0, type=int)
        parser.add_argument("--verbose", action='store_true', default=False)
        parser.add_argument("--show-total", action='store_true', default=False)
    
    def handle(self, *args, **options):
        limit = options.get('limit')
        verbose = options.get('verbose')
        show_total = options.get('show_total')
        loan_dataset = utils.load_loan_data(limit=limit, verbose=verbose)
        loan_new = [Loan(**x) for x in loan_dataset]
        loan_bulk = Loan.objects.bulk_create(loan_new)
        print(f"New Loans: {len(loan_bulk)}")
        if show_total:
            print(f"Total Loans: {Loan.objects.count()}")
