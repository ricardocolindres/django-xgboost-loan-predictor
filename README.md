# Introduction
The Loan Predictor System was designed to assist public credit agents in the process of analyzing and assessing the risk associated with loans secured by the Small Business Association (SBA) of the United States of America. The SBA is a government agency created in 1954 to promote loans and financial support to small businesses by partially securing the funds' private banks disburse to these businesses. As one can imagine, the credit agents commissioned with the task of evaluating the risk associated with every single application submitted can easily be overwhelmed by the sheer volume of submissions. From 1986 to 2014, the agency received almost one million submissions. The data associated with the loans disbursed in this time frame can be accessed at https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied where each loan contains twenty-seven (27) different features describing each data point. The purpose of this project is to build a system that employs state-of-the-art machine learning models to automatically classify new loan applications as low- or high-risk. Moreover, the systems should provide the agents with the probability associated with any given loan belonging to any of the mentioned categories (i.e., low- or high-risk). To make the system even more powerful, I have added one more functionality. Once an application is submitted and classified as low risk (only for low-risk loans), the system will create a risk assessment that includes the suggestion of newly optimized parameters for the term (i.e., the period to repay the loan) and amount of money to be disbursed for the requested loan to reduce the risk of the business defaulting. Consequently, the SBA Loan Predictor will allow public credit agents to efficiently manage their time by automatically classifying new applications as low or high risk and focusing their valuable time on assessing those loan applications that will yield the best results for all parties involved. Furthermore, the system tracks new applications and their outcomes or accepts new loan data to periodically retrain the ML model and keep predictions and assessments accurate throughout time. Administrators and credit agents can easily log in or register using the user-friendly forms provided. These forms use the robust authentication systems included in Django to securely allow users to access the systems according to their respective permissions. On the other hand, prospective clients can easily submit new applications through a RESTFUL API and an intuitive user interface.

![signin](https://github.com/ricardocolindres/django-xgboost-loan-predictor/assets/83890387/f4948871-023c-4a6c-b36e-887df09d16d6)

