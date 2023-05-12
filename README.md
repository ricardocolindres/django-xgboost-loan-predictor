# Introduction
The Loan Predictor System was designed to assist public credit agents in the process of analyzing and assessing the risk associated with loans secured by the Small Business Association (SBA) of the United States of America. The SBA is a government agency created in 1954 to promote loans and financial support to small businesses by partially securing the funds private banks disburse to these businesses. As one can imagine, the credit agents commissioned with the task of evaluating the risk associated with every single application submitted can easily be overwhelmed by the sheer volume of submissions. From 1986 to 2014, the agency received almost one million submissions. The data associated with the loans disbursed in this time frame can be accessed at https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied where each loan contains twenty-seven (27) different features describing each data point. The purpose of this project is to build a system that employs state-of-the-art machine learning models to automatically classify new loan applications as low- or high-risk. Moreover, the systems should provide the agents with the probability associated with any given loan belonging to any of the mentioned categories (i.e., low- or high-risk). To make the system even more powerful, I have added one more functionality. Once an application is submitted and classified as low risk (only for low-risk loans), the system will create a risk assessment that includes the suggestion of newly optimized parameters for the term (i.e., the period to repay the loan) and amount of money to be disbursed for the requested loan to reduce the risk of the business defaulting. Consequently, the SBA Loan Predictor will allow public credit agents to efficiently manage their time by automatically classifying new applications as low or high risk and focusing their valuable time on assessing those loan applications that will yield the best results for all parties involved. Furthermore, the system tracks new applications and their outcomes or accepts new loan data to periodically retrain the ML model and keep predictions and assessments accurate throughout time. Administrators and credit agents can easily log in or register using the user-friendly forms provided. These forms use the robust authentication systems included in Django to securely allow users to access the systems according to their respective permissions. On the other hand, prospective clients can easily submit new applications through a RESTFUL API and an intuitive user interface.

The process of creating the actual model can be reviewed at: https://www.kaggle.com/code/ricardocolindres/loan-default-prediction-loan-parameter-optimizer

![signin](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/f4948871-023c-4a6c-b36e-887df09d16d6)

# Installation

1.	Clone the repository to your preferred location. 
2.	Create a new virtual environment using the requirements contained in the requirements.txt file. I personally like using pipenv so you will also find a Pipifile in the repository. You could also use this file to load your virtual environment. Once your virtual environment is ready, please activate it in your favorite IDE, for me, this is VS code. 
3.	Download the “cold_start.csv” file from the following link: https://www.dropbox.com/s/d7unt2wqm7qluof/cold_load.csv?dl=0
4.	This CSV file contains all loans the SBA has disbursed from 1986 to 2014. The data has been cleaned and prepared for the ML Model the system will be running. This file MUST be placed on the data folder contained within the loanpredictor directory unless you change the DATA_DIR variable in the settings.py file to declare a new path for the data. As the name suggests, “cold start”, this data is essential to initiate the system. 

```
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Cold start data path
DATA_DIR = os.path.join(BASE_DIR, 'data')
```
5.	Before the server and other services run, we’ve got to modify some files. This project was built using the open-source templates created by Creative Tim at https://github.com/app-generator/django-admin-soft-dashboard. These templates were adapted to fit the purpose of this particular project; thus, some changes have been made to this library. When you created your virtual environment, the following package should have been installed: django-admin-soft-dashboard==1.0.12. This package should live within the directory that contains your virtual environment. In Windows, it should look something like this: C:\Users\YourUser\.virtualenvs\YourVirtualEnviorment\Lib\site-packages\admin_soft. Once you have accessed the directory, please replace the files forms.py and urls.py with the ones contained in the repository in the data folder within the loanpredictor directory. You can certainly add the admin_soft folder as an independent app directly into the Django project; however, this will require additional configurations that, in this case, are not worth the time since the changes are so little. As you can see, the static and template folders, which were originally contained in this same package’s directory, were copied to the main project’s directory. The files here were heavily modified; thus, having them here makes sense. 
6.	Assuming you have Docker Container installed on your computer, please run the docker-compose file. Doing so will start the following services: redis, PostgreSQL, and PgAdmin (database interface). Please keep in mind that redis will be later communicating with Celery to automate some tasks. Celcery has dropped support for Windows, so, if you are using a PC, will need to run your code within a virtual environment. Please refer to this article if you need help doing this: https://www.codedisciples.in/celery-windows.html  
7.	All the settings for these docker containers should match with the ones in the settings.py file. However, if you decide to change some of the settings in the docker-compose file, don’t forget to update them in your settings.py file. Now you can run the following commands in your terminal (your virtual environment should be activated, and you should be inside the loanpredictor directory). Run each command at a time. 
```
# Make migrations
python manage.py makemigrations
# Migrate to database
python manage.py migrate
# Cold Start. Load initial data
python manage.py loader --verbose --show-total
#Create Super User, Follow Instructions. 
python manage.py createsuperuser

```
8.	Now, we can start the celery workers. If you would like to adjust the schedule in which the task will be run, please do so in the celery.py file.
```
# Start workers and beat schedule
celery -A loanpredictor worker -l info --beat

``` 
9.	Run Django Server
```
# Run Django server
python manage.py runserver
```
10.	Log in into the administrator account using the super user you just created. Finally, create a group for credit agents with permissions to modify and create all the loans and risk assessments tables. You are all set!  

# The Application and The XG-Boost Model
As previously discussed, the main purpose of this application is to assist public credit agents. At the core of the application is a powerful XG Boost ML model that makes accurate predictions regarding new loan applications. This ML model and other services are made available to clients using the Django Web Framework. As seen in the Conceptual Architecture Diagram, Django is central to all functionalities. Moreover, when building the application, it was particularly important for me to integrate a solid ML lifecycle workflow into the application. As one can see, Jupyter Notebooks are directly integrated into Django, which enables data exploration and analysis using Django’s easy-to-use ORM (Object Relational Mapper). This allows data scientists, data engineers, and ML engineers to constantly experiment with data and tweak the ML models as they see fit. To further contribute to a robust ML lifecycle, I have automated the training of the ML model using Redis and Celery. Every 24 hours (this could be adjusted to any desired period), the celery’s workers query the database for loans that meet certain criteria and retrain the model based on this newly acquired data. A new model is created, and the previous ones are deactivated. The administrators can easily manage all the models and compare each one’s performance, thus allowing them to put any of the available models into production (by default, the latest one goes into production). The platforms also keep track of all the models’ parameters. The models and the associated data are all saved into a central model registry in a PostgreSQL database for easy experimentation, reproducibility, deployment, and analysis. The database, of course, serves other purposes, such as storing all loans, risk assessments, permissions, and many other permanent data storage needs. Finally, the clients can easily submit new applications using a RESTFUL API. Every time a POST request comes through, the applications generate a new entry in the loan tables, and if the loan is classified as low risk, it also creates a risk assessment. The credit agents can then access all the loan applications from their staff portal and filter according to risk. From here, they can easily approve and keep track of the default status of any loan. After one year from the loans reaching their maturity date (i.e., the day on which the loans were supposed to be paid in full), the system will automatically disable updated and use these loans as new data to feed the ML model and retrain it.

![model_diagram](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/72ade4ba-fa26-483c-8331-d448d43668bf)

# The User Interface
In the following section, I will quickly guide you through the interface and what is going on under the hood. When a staff member or administrator visits the home page, they will be greeted with the following page: 

![home](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/5e8689a5-3e8b-4c8d-b734-c35df7b1aa71)

From this point, the staff and administrator can use the sign-in and registration forms to access all the application’s features. Under the hood, the powerful Django authentication system has been used. All newly registered users will have no permissions set by default; therefore, they won’t be able to see anything but their profile. The administrators can change these permissions. If a user is a credit agent, it should be added to the credit agent group created at installation, which gives permissions for modifying the loan and risk assessment. Of course, a new administrator can be added similarly.

![signin](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/756a7b42-a25e-4310-afb2-65346dd691dd)
![register](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/085c5784-72a2-4912-9af9-72bf129e2246)
![agent_profile](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/a775f214-cba2-4331-a0c8-b5996ae80c56)

Going forward, I will only use the administrator portal as an example. Keep in mind that the credit agent will have the same access to the loan and risk assessment interface and will not be able to see any other data concerning the applications. Let's begin with the tables mentioned. From the loan interface, the user can access all loans and applications. An application is not necessarily a loan, a loan is only a loan if it has the approval data and approved attributes filled out. Loans that have not reached one year after maturity and have not been defaulted on will appear as active and will not be used to train the ML model. On the other hand, loan applications that are evaluated as high risk by the ML model will automatically be set as inactive, while those considered low risk will be active, and additional risk assessment data will be appended to them. This risk assessment data includes optimized parameters for the term and amount of money requested. The risk assessment gives the agents the probability that the client will default on the loan based on the requested parameters and the optimized ones. From this point, the credit agent can approve the loan by adding approval data or reject it by setting it as inactive. As you can see, the agent can filter and search the loans easily. They can also, at any point, set a active loan as defaulted. Finally, batches of loans can be added directly to the database, but no risk assessment will be run on them. Only individual loans coming from the client’s side will be assessed.

![loan_admin](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/d37500bc-2ba7-43a2-a7da-a2f65a45c9e1)
![loan](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/7322aac0-5dc7-4feb-93b1-8e8b4e880229)
![risk_assesmnet](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/2ebef1ae-42a0-410c-981d-c08f9ea40df0)

Moving on, let's explore how I’ve designed the ML lifecycle. Since the ML model is at the core of this application, having a proper way to experiment, reproduce, deploy, and manage the models and components is paramount. Of course, there are many robust open-source platforms out there, such as ML Flow, that can accomplish all these goals. However, my objective was to completely integrate the ML lifecycle workflow into my web applications and customize it according to requirements. Moreover, this allows for very powerful integration of the model into the logic of the application. For example, delivering extra value by not just categorizing loans but also optimizing their parameters. 

![models_admin](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/6fda6f22-182e-4378-b97a-e1e3f35037de)
![Untitled-1](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/fb31ea84-eb64-4db8-82b2-68c9e940ca30)

Consequently, the application uses special queries to request data from the database and automatically trains and deploys new models. Also, Jupyter Notebooks are directly integrated into Django and tis easy-to-use ORM (Object Relational Mapper).
```
# Jupyter Notebooks Integration
import os, sys
PWD = os.path.dirname(os.getcwd())

PROJ_MISSING_MSG = """Set an enviroment variable:\n
`DJANGO_PROJECT=your_project_name`\n
or call:\n
`init_django(your_project_name)`
"""

def init_django(project_name=None):
    os.chdir(PWD)
    project_name = project_name or os.environ.get('DJANGO_PROJECT') or None
    if project_name == None:
        raise Exception(PROJ_MISSING_MSG)
    sys.path.insert(0, PWD)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', f'{project_name}.settings')
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    import django
    django.setup()
```
As a result, parameters can be easily tweaked by engineers to improve the training process. For administrators, the lifecycle of the models is even easier. I have automated the training of the ML model using Redis and Celery. Users can easily create new training schedules or manage existing ones using a user-friendly interface. Under the hood, the model will be retrained according to all the parameters set. 

```
# Custom Query for training the ML model on data contained in the database

class MlModelQuerySet(models.QuerySet):
    def ml_data(self):
        return self.filter(maturity_date__lt=AVAIALBLE_TRUE_DATA)
    
class MlModelManager(models.Manager):
    def get_queryset(self, *args, **kwargs):
        return MlModelQuerySet(self.model, using=self._db)
    
    def ml_data(self):
        return self.get_queryset().ml_data()
        
# Function fro training model

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
        

# Function for optimizing parameters

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


```

The administrators can easily manage and compare all the models and put the best one into production. All models are saved into a central model registry in PostgreSQL. Finally, there is also another interface to review the results of the tasks automated by Celery. This will let the user know everything about how any given task was executed. For this application, Celery runs model training every 24 hours and checks active loans to decide whether they should be set as inactive based on the conditions we have previously discussed. All the data in PostgreSQL can be easily visualized using PgAdmin.


```
# Celery Schudule
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
```


![tasks](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/ae2ef222-f064-4289-b681-6203c06ae903)
![celery_taks](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/2299a154-eafe-44f5-acef-5a6c2fe2d574)

Finally, let’s explore how clients submit applications. Different endpoints have been defined to communicate with the server through a RESTFUL API, specifically the Django Rest Framework. For submitting a new loan application, a client can visit the endpoint /ebanking/apply. This endpoint only allows post requests. In this case, I do not want to expose the risk assessment to clients; therefore, I do not return any probabilities, assessments, or data at all. However, serialized information about the loans can be accessed at the endpoint /ebanking/loan and ebanking/loan/<int:id>. I set these endpoints up just in case some sort of front-end service would be preferred for credit agents. 

![application](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/99ad258e-69c2-4ea1-a24c-512d79aa7381)
![API](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/405df8e8-ae6d-4f9a-8b39-b9182cd41b91)


*Disclosure: This project WAS NOT BEEN COMMISIONED by the Small Business Association (SBA) of the United States of America. It is an independent project built using a publicly available real dataset containing all the loans this agency has disbursed from 1886 to 2014. 

FOR MORE INFORMATION OR HELP PLEASE CONTACT ME AT RICARDOCOLINDRES@ME.com

