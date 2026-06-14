from django.urls import path
from . import views

urlpatterns = [
    path('ask/', views.ask, name='ask'),
    path('health/', views.health, name='health'),
]