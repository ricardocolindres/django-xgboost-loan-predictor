from django.urls import path
from . import views

urlpatterns = [
    path('loans/', views.LoanList.as_view()),
    path('loans/<int:id>/', views.LoanDetails.as_view()),
    path('apply/', views.LoanCreate.as_view()),
]