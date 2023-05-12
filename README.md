# Introduction
The Loan Predictor System was designed to assist public credit agents in the process of analyzing and assessing the risk associated with loans secured by the Small Business Association (SBA) of the United States of America. The SBA is a government agency created in 1954 to promote loans and financial support to small businesses by partially securing the funds' private banks disburse to these businesses. As one can imagine, the credit agents commissioned with the task of evaluating the risk associated with every single application submitted can easily be overwhelmed by the sheer volume of submissions. From 1986 to 2014, the agency received almost one million submissions. The data associated with the loans disbursed in this time frame can be accessed at https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied where each loan contains twenty-seven (27) different features describing each data point. The purpose of this project is to build a system that employs state-of-the-art machine learning models to automatically classify new loan applications as low- or high-risk. Moreover, the systems should provide the agents with the probability associated with any given loan belonging to any of the mentioned categories (i.e., low- or high-risk). To make the system even more powerful, I have added one more functionality. Once an application is submitted and classified as low risk (only for low-risk loans), the system will create a risk assessment that includes the suggestion of newly optimized parameters for the term (i.e., the period to repay the loan) and amount of money to be disbursed for the requested loan to reduce the risk of the business defaulting. Consequently, the SBA Loan Predictor will allow public credit agents to efficiently manage their time by automatically classifying new applications as low or high risk and focusing their valuable time on assessing those loan applications that will yield the best results for all parties involved. Furthermore, the system tracks new applications and their outcomes or accepts new loan data to periodically retrain the ML model and keep predictions and assessments accurate throughout time. Administrators and credit agents can easily log in or register using the user-friendly forms provided. These forms use the robust authentication systems included in Django to securely allow users to access the systems according to their respective permissions. On the other hand, prospective clients can easily submit new applications through a RESTFUL API and an intuitive user interface.

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
10.	Log in into the administrator account using the super user you just created. Finally, create a group for credit agents and permissions to modify and create all the loans and risk assessments tables. You are all set!  
