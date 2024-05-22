from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("result/", views.result, name="result"),
    path("admin/", views.index_admin, name="index_admin"),
    path("history/", views.prediction_history, name='prediction_history'),

]

