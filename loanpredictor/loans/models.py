import pickle
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from dateutil.relativedelta import relativedelta
from datetime import datetime, date
from ml.utlis import load_model, optimize_loan_parameters, prepare_data
from django.dispatch import receiver
from django.db.models.signals import post_save, pre_save
import pandas as pd

# Loans will remain active after 1 year of reaching maturity
TIME_AFTER_MATURITY_ACTIVE = 12
# Filter Loans only before 2014 which is the data avialable in the dataset
AVAIALBLE_TRUE_DATA = datetime.strptime('2014-12-31', '%Y-%m-%d')

class EconomyChoice(models.IntegerChoices):
    Agriculture = 0, _('Agriculture, Forestry, Fishing and Hunting')
    Mining = 1, _('Mining, Quarrying, and Oil and Gas Extraction')
    Utilities = 2, _('Utilities')
    Construction = 3, _('Construction')
    Manufacturing = 4, _('Manufacturing')
    Wholesale = 5, _('Wholesale Trade')
    Retail = 6, _('Retail Trade')
    Transportation = 7, _('Transportation and Warehousing')
    Information = 8, _('Information')
    Finance = 9, _('Finance and Insurance')
    RealEstate = 10, _('Real Estate and Rental and Leasing')
    ProfessionalServices = 11, _('Professional, Scientific, and Technical Services')
    Managment = 12, _('Management of Companies and Enterprises')
    AdministrativeServices = 13, _('Administrative and Support and Waste Management and Remediation Services')
    EducationalServices = 14, _('Educational Services')
    HealthCare = 15, _('Health Care and Social Assistance')
    Arts = 16, _('Arts, Entertainment, and Recreation')
    AcommodationFood = 17, _('Accommodation and Food Services')
    Others = 18, _('Other Services (Except Public Administration)')
    PublicAdministration = 19, _('Public Administration')

class StateChoice(models.IntegerChoices):
    AK = 0, _('Alaska(AK)')
    AL = 1, _('Alabama(AL)')
    AR = 2, _('Arkansas(AR)')
    AZ = 3 , _('Arizona(AZ)')
    CA = 4, _('California(CA)')
    CO = 5, _('Colorado(CO)')
    CT = 6, _('Connecticut(CT)')
    DC = 7, _('District of Columbia(DC)')
    DE = 8, _('Delaware(DE)')
    FL = 9, _('Florida(FL)')
    GA = 10, _('Georgia(GA)')
    HI = 11, _('Hawaii(HI)')
    IA = 12, _('Iowa(IA)')
    ID = 13, _('Idaho(ID)')
    IL = 14, _('Illinois(IL)')
    IN = 15, _('Indiana(IN)')
    KS = 16, _('Kansas(KS)')
    KY = 17, _('Kentucky(KY)')
    LA = 18, _('Louisiana(LA)')
    MA = 19, _('Massachusetts(MA)')
    MD = 20, _('Maryland(MD)')
    ME = 21, _('Maine(ME)')
    MI = 22, _('Michigan(MI)')
    MN = 23, _('Minnesota(MN)')
    MO = 24, _('Missouri(MO)')
    MS = 25, _('Mississippi(MS)')
    MT = 26, _('Montana(MT)')
    NC = 27, _('North Carolina(NC)')
    ND = 28, _('North Dakota(ND)')
    NE = 29, _('Nebraska(NE)')
    NH = 30, _('New Hampshire(NH)')
    NJ = 31, _('New Jersey(NJ)')
    NM = 32, _('New Mexico(NM)')
    NV = 33, _('Nevada(NV)')
    NY = 34, _('New York(NY)')
    OH = 35, _('Ohio(OH)')
    OK = 36, _('Oklahoma(OK)')
    OR = 37, _('Oregon(OR)')
    PA = 38, _('Pennsylvania(PA)')
    RI = 39, _('Rhode Island(RI)')
    SC = 40, _('South Carolina(SC)')
    SD = 41, _('South Dakota(SD)')
    TN = 42, _('Tennessee(TN)')
    TX = 43, _('Texas(TX)')
    UT = 44, _('Utah(UT)')
    VA = 45, _('Virginia(VA)')
    VT = 46, _('Vermont(VT)')
    WA = 47, _('Washington(WA)')
    WI = 48, _('Wisconsin(WI)')
    WV = 49, _('West Virginia(WV)')
    WY = 50, _('Wyoming(WY)')

class RiskChoice(models.IntegerChoices):
    High = 1, _('High Risk / Not Approved')
    Low = 0, _('Low Risk / Approved')

class Name(models.CharField):
    def __init__(self, *args, **kwargs):
        super(Name, self).__init__(*args, **kwargs)

    def get_prep_value(self, value):
        return str(value).upper()


class MlModelQuerySet(models.QuerySet):
    def ml_data(self):
        return self.filter(maturity_date__lt=AVAIALBLE_TRUE_DATA)
    
class MlModelManager(models.Manager):
    def get_queryset(self, *args, **kwargs):
        return MlModelQuerySet(self.model, using=self._db)
    
    def ml_data(self):
        return self.get_queryset().ml_data()

class Loan(models.Model):
    # Being super extrict with null values
    loan_id = models.BigAutoField(primary_key=True)
    internal_id = models.BigIntegerField(null=True, blank=True)
    bs_name = Name(null=False, blank=False, max_length=255, verbose_name='Business Name')
    email = models.EmailField(null=True, blank=True, default=None)
    phone = models.BigIntegerField(null=True, blank=True, default=None)
    approval_date = models.DateField(null=True, blank=True, auto_now=False, auto_now_add=False)
    maturity_date = models.DateField(null=True, blank=True, auto_now=False, auto_now_add=False)
    state = models.IntegerField(null=False, blank=False, choices=StateChoice.choices)
    term = models.IntegerField(null=False, blank=False)
    no_emp = models.IntegerField(null=False, blank=False, verbose_name='Number of Current Employees')
    created_jobs = models.IntegerField(null=False, blank=False, verbose_name='Number of Created Jobs')
    retained_jobs = models.IntegerField(null=False, blank=False, verbose_name='Number of Retained Jobs')
    gross_appv = models.DecimalField(decimal_places=2, max_digits=20, blank=False, null=False)
    recession = models.BooleanField(null=False, blank=False, verbose_name='Recession likely (or active) during loan?')
    secured_loan = models.BooleanField(null=False, blank=False, verbose_name='Is the loan secured?')
    gov_secured =  models.DecimalField(decimal_places=2, max_digits=5, blank=False, null=False, verbose_name='Goverment Secured Loan Percentage')
    is_rural = models.BooleanField(null=False, blank=False, verbose_name='Is the business rural?')
    low_doc = models.BooleanField(null=False, blank=False, verbose_name='Low Docuemnts Program?')
    new_business = models.BooleanField(null=False, blank=False, verbose_name='Is the business new?')  
    econ_sector = models.IntegerField(null=False, blank=False, choices=EconomyChoice.choices, verbose_name='Economy Sector')
    inflation_on_loan = models.BooleanField(null=False, blank=False, verbose_name='Unusual inflation likely (or active) during loan? (>4)')
    unemployment_on_loan = models.BooleanField(null=False, blank=False, verbose_name='Unusual unemployment likely (or active) during loan? (>6)')
    defaulted = models.BooleanField(null=False, blank=False, default= False)
    active = models.BooleanField(null=False, blank=False, default=True)
    approved = models.BooleanField(null=False, blank=False, default=False)
    last_updated = models.DateTimeField(auto_now=True)
    objects = MlModelManager()

    def __str__(self):
        return f"Loan #{self.internal_id}"

    class Meta:
        ordering = ['-approval_date']

    # explain bulck and save
    def save(self, *args, **kwargs):
        #Not all loans must be approved
        if self.approval_date:
            self.approved =True
            now = date.today()
            self.maturity_date = self.approval_date + relativedelta(months=+self.term)
            active_frame = self.maturity_date + relativedelta(months=+TIME_AFTER_MATURITY_ACTIVE)
            if active_frame < now:
                self.active = False
            if self.defaulted:
                self.active = False                

        super().save(*args, **kwargs)

    def update(self, *args, **kwargs):
        #Not all loans must be approved
        if self.approval_date:
            self.approved =True
            now = date.today()
            self.maturity_date = self.approval_date + relativedelta(months=+self.term)
            active_frame = self.maturity_date + relativedelta(months=+TIME_AFTER_MATURITY_ACTIVE)
            if active_frame < now:
                self.active = False
        super().save(*args, **kwargs)
    
class RiskAssessement(models.Model):
    loan = models.OneToOneField(Loan, primary_key=True, on_delete=models.CASCADE)
    default_risk = models.IntegerField(null=False, blank=False, choices=RiskChoice.choices)
    default_prob = models.FloatField(blank=True, null=False, verbose_name='Probabilty loan is defaulted')
    optimized_prob = models.FloatField(blank=True, null=True, verbose_name='Probabilty loan is defaulted with optimized parameters ')
    optimized_term = models.IntegerField(null=True, blank=False)
    optimized_gross_appv = models.DecimalField(decimal_places=2, max_digits=20, blank=True, null=True)


@receiver(post_save, sender=Loan, dispatch_uid="risk_assestment")
def model_post_save(sender, created, instance, *args, **kwargs):
        if created: 
            if not instance.defaulted and not instance.approved:
                #Create Risk Assesment    
                model = load_model()
                loan_data = {'state': instance.state, 'term': instance.term, 'no_emp':instance.no_emp, 'created_jobs':instance.created_jobs,
                            'retained_jobs':instance.retained_jobs, 'gross_appv':instance.gross_appv, 'recession': instance.recession, 
                            'secured_loan':instance.secured_loan, 'gov_secured':instance.gov_secured,'is_rural':instance.is_rural, 
                            'low_doc':instance.low_doc, 'new_business':instance.new_business, 'econ_sector':instance.econ_sector,
                            'inflation_on_loan':instance.inflation_on_loan,'unemployment_on_loan':instance.unemployment_on_loan}
                
                df = pd.DataFrame(loan_data, index=[0])
                df = prepare_data(df)
                # get risk
                risk = model.predict(df)[0]
                #((271000, 84), 0.99368334, 0.9854964, 0.008186936) return structure
                # Only Run parameter optimization for low risk loans
                if risk == 0:
                    optimized_params = optimize_loan_parameters(df=df, bst=model)
                # If loan is High Risk, do not run optimization
                else:
                    optimized_params = None
                # If optimized parameters exist
                if optimized_params:
                    risk_prob = model.predict_proba(df)[0][1]
                    optimized_risk_prob = risk_prob - optimized_params[3]
                    risk_assesment = RiskAssessement.objects.create(
                                                        loan_id = instance.loan_id,
                                                        default_risk = risk,
                                                        default_prob = round(risk_prob*100, 2),
                                                        optimized_prob = round(optimized_risk_prob*100, 2),
                                                        optimized_term = optimized_params[0][1],
                                                        optimized_gross_appv = optimized_params[0][0],)
                else:
                    risk_prob = model.predict_proba(df)[0][1]
                    risk_assesment = RiskAssessement.objects.create(
                                                        loan_id = instance.loan_id,
                                                        default_risk = risk,
                                                        default_prob = round(risk_prob *100, 2),
                                                        optimized_prob = None,
                                                        optimized_term = None,
                                                        optimized_gross_appv = None,)
                    
                if risk == 1: 
                    obj = Loan.objects.filter(loan_id=instance.loan_id).update(active=False)


    