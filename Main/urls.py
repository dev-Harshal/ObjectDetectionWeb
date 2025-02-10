from django.urls import path
from Main.views import index_view

urlpatterns = [
    path('', index_view, name='index-view')
]
