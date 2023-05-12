from rest_framework import serializers
from .models import Loan, RiskAssessement


class LoanSerializer(serializers.ModelSerializer):
    gross_secured = serializers.SerializerMethodField(method_name='calculate_gross_secured')
    loan_amount_requested = serializers.DecimalField(decimal_places=2, max_digits=20, source='gross_appv')
    class Meta:
        model = Loan
        fields = ["bs_name", 'email', 'phone',
                  "state","term","no_emp","created_jobs",
                  "retained_jobs", 'loan_amount_requested', "recession","secured_loan",
                  "gov_secured","is_rural","low_doc","new_business",
                  "econ_sector","inflation_on_loan","unemployment_on_loan", "gross_secured"]
        
    def calculate_gross_secured(self, loan:Loan):
        return loan.gross_appv * (loan.gov_secured/100)
    
    def validate(self, attrs):
        return super().validate(attrs)
 