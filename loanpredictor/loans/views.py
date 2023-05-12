from django.shortcuts import get_object_or_404
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework import status
from loans.models import Loan
from .models import Loan
from .serializers import LoanSerializer

# Create your views here.

class LoanList(ListAPIView):
    queryset =  Loan.objects.all()
    serializer_class = LoanSerializer
    
    def get_serializer_context(self):
        return {'request': self.request}
    
    pagination_class = PageNumberPagination


class LoanCreate(CreateAPIView):
    queryset =  Loan.objects.all()
    serializer_class = LoanSerializer

    def get_serializer_context(self):
        return {'request': self.request}
    
class LoanDetails(APIView):
    def get(self, request, id):
        loan = get_object_or_404(Loan, pk=id)
        serializer = LoanSerializer(loan)
        return Response(serializer.data)


