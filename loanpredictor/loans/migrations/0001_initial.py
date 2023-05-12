# Generated by Django 4.2.1 on 2023-05-11 02:47

from django.db import migrations, models
import django.db.models.deletion
import loans.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Loan',
            fields=[
                ('loan_id', models.BigAutoField(primary_key=True, serialize=False)),
                ('internal_id', models.IntegerField(blank=True, null=True)),
                ('bs_name', loans.models.Name(max_length=255, verbose_name='Business Name')),
                ('email', models.EmailField(default=None, max_length=254, null=True)),
                ('phone', models.IntegerField(default=None, null=True)),
                ('approval_date', models.DateField(blank=True, null=True)),
                ('maturity_date', models.DateField(blank=True, null=True)),
                ('state', models.IntegerField(choices=[(0, 'Alaska(AK)'), (1, 'Alabama(AL)'), (2, 'Arkansas(AR)'), (3, 'Arizona(AZ)'), (4, 'California(CA)'), (5, 'Colorado(CO)'), (6, 'Connecticut(CT)'), (7, 'District of Columbia(DC)'), (8, 'Delaware(DE)'), (9, 'Florida(FL)'), (10, 'Georgia(GA)'), (11, 'Hawaii(HI)'), (12, 'Iowa(IA)'), (13, 'Idaho(ID)'), (14, 'Illinois(IL)'), (15, 'Indiana(IN)'), (16, 'Kansas(KS)'), (17, 'Kentucky(KY)'), (18, 'Louisiana(LA)'), (19, 'Massachusetts(MA)'), (20, 'Maryland(MD)'), (21, 'Maine(ME)'), (22, 'Michigan(MI)'), (23, 'Minnesota(MN)'), (24, 'Missouri(MO)'), (25, 'Mississippi(MS)'), (26, 'Montana(MT)'), (27, 'North Carolina(NC)'), (28, 'North Dakota(ND)'), (29, 'Nebraska(NE)'), (30, 'New Hampshire(NH)'), (31, 'New Jersey(NJ)'), (32, 'New Mexico(NM)'), (33, 'Nevada(NV)'), (34, 'New York(NY)'), (35, 'Ohio(OH)'), (36, 'Oklahoma(OK)'), (37, 'Oregon(OR)'), (38, 'Pennsylvania(PA)'), (39, 'Rhode Island(RI)'), (40, 'South Carolina(SC)'), (41, 'South Dakota(SD)'), (42, 'Tennessee(TN)'), (43, 'Texas(TX)'), (44, 'Utah(UT)'), (45, 'Virginia(VA)'), (46, 'Vermont(VT)'), (47, 'Washington(WA)'), (48, 'Wisconsin(WI)'), (49, 'West Virginia(WV)'), (50, 'Wyoming(WY)')])),
                ('term', models.IntegerField()),
                ('no_emp', models.SmallIntegerField(verbose_name='Number of Current Employees')),
                ('created_jobs', models.SmallIntegerField(verbose_name='Number of Created Jobs')),
                ('retained_jobs', models.SmallIntegerField(verbose_name='Number of Retained Jobs')),
                ('gross_appv', models.DecimalField(decimal_places=2, max_digits=20)),
                ('recession', models.BooleanField(verbose_name='Recession likely (or active) during loan?')),
                ('secured_loan', models.BooleanField(verbose_name='Is the loan secured?')),
                ('gov_secured', models.DecimalField(decimal_places=2, max_digits=5, verbose_name='Goverment Secured Loan Percentage')),
                ('is_rural', models.BooleanField(verbose_name='Is the business rural?')),
                ('low_doc', models.BooleanField(verbose_name='Low Docuemnts Program?')),
                ('new_business', models.BooleanField(verbose_name='Is the business new?')),
                ('econ_sector', models.SmallIntegerField(choices=[(0, 'Agriculture, Forestry, Fishing and Hunting'), (1, 'Mining, Quarrying, and Oil and Gas Extraction'), (2, 'Utilities'), (3, 'Construction'), (4, 'Manufacturing'), (5, 'Wholesale Trade'), (6, 'Retail Trade'), (7, 'Transportation and Warehousing'), (8, 'Information'), (9, 'Finance and Insurance'), (10, 'Real Estate and Rental and Leasing'), (11, 'Professional, Scientific, and Technical Services'), (12, 'Management of Companies and Enterprises'), (13, 'Administrative and Support and Waste Management and Remediation Services'), (14, 'Educational Services'), (15, 'Health Care and Social Assistance'), (16, 'Arts, Entertainment, and Recreation'), (17, 'Accommodation and Food Services'), (18, 'Other Services (Except Public Administration)'), (19, 'Public Administration')], verbose_name='Economy Sector')),
                ('inflation_on_loan', models.BooleanField(verbose_name='Unusual inflation likely (or active) during loan? (>4)')),
                ('unemployment_on_loan', models.BooleanField(verbose_name='Unusual unemployment likely (or active) during loan? (>6)')),
                ('defaulted', models.BooleanField(default=False)),
                ('active', models.BooleanField(default=False)),
                ('approved', models.BooleanField(default=False)),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-approval_date'],
            },
        ),
        migrations.CreateModel(
            name='RiskAssessement',
            fields=[
                ('loan', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='loans.loan')),
                ('default_risk', models.IntegerField(choices=[(1, 'High Risk / Not Approved'), (0, 'Low Risk / Approved')])),
                ('default_prob', models.FloatField(blank=True, verbose_name='Probabilty loan is defaulted')),
                ('optimized_prob', models.FloatField(blank=True, null=True, verbose_name='Probabilty loan is defaulted with optimized parameters ')),
                ('optimized_term', models.IntegerField(null=True)),
                ('optimized_gross_appv', models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True)),
            ],
        ),
    ]
