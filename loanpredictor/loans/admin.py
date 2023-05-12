from typing import Any, List, Optional, Tuple
from django.contrib import admin
from django.db.models.query import QuerySet
from . import models

class RiskFilter(admin.SimpleListFilter):
    title = 'Risk'
    parameter_name = 'risk'
    def lookups(self, request: Any, model_admin: Any) -> List[Tuple[Any, str]]:
        return [
            ('1', 'High Risk'),
            ('0', 'Low Risk')
        ]
    
    def queryset(self, request: Any, queryset: QuerySet[Any]) -> QuerySet[Any] | None:
        if self.value() == '1':
            return queryset.select_related('riskassessement').filter(riskassessement__default_risk = 1)
        elif self.value() == '0':
            return queryset.select_related('riskassessement').filter(riskassessement__default_risk = 0)
        
class RiskInline(admin.StackedInline):
    model = models.RiskAssessement
    readonly_fields = ['loan_id', 'default_risk', 'default_prob',
                    'optimized_prob','optimized_term', 'optimized_gross_appv']

# Register your models here.
@admin.register(models.Loan)
class LoanAdmin(admin.ModelAdmin):
    inlines = [RiskInline, ]
    fields = ['internal_id', 'bs_name', 'email', 'phone', 'approval_date', 'maturity_date', 'gross_appv', 'term', 'state', 
              'econ_sector','no_emp','created_jobs','retained_jobs', 'gov_secured',
              'new_business','is_rural','low_doc','secured_loan',
              'recession','inflation_on_loan','unemployment_on_loan', 'defaulted', 'approved']
    
    
    list_display = ['loan_id', 'bs_name', 'internal_id', 'term', 'approval_date',  'state', 'defaulted', 'active', 'risk', 'probability_of_default']
    list_filter = ['approval_date', RiskFilter]
    list_select_related = ['riskassessement']
    readonly_fields = ['maturity_date']
    search_fields = ['internal_id', 'bs_name']

    def has_change_permission(self, request, obj=None):
        if obj is not None and obj.active == False:
            return False
        return super().has_change_permission(request, obj=obj)
    
    def risk(self, loan):
        risk = loan.riskassessement.default_risk
        if risk == 1:
            return 'High Risk'
        else:
            return 'Low Risk'
        
    def probability_of_default(self, loan):
        return loan.riskassessement.default_prob
    
@admin.register(models.RiskAssessement)
class LoanAdmin(admin.ModelAdmin):
    fields = ['loan_id', 'default_risk', 'default_prob',
                    'optimized_prob','optimized_term', 'optimized_gross_appv']
    list_display = ['loan_id', 'default_risk', 'default_prob',
                    'optimized_prob','optimized_term', 'optimized_gross_appv']
    
    readonly_fields = ['loan_id']
