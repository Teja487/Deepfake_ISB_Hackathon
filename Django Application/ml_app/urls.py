from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views


app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', views.entry_page, name='home'),
    path('index/', views.index, name='index'),
    path('image_index/', views.image_index, name='image_index'),
    path('image_predict/', views.image_predict, name='image_predict'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict_page, name='predict'),
    path('cuda_full/', views.cuda_full, name='cuda_full'),
    path('text_upload/', views.text_upload, name='text_upload'),
    path('text_predict/', views.text_predict, name='text_predict'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.IMAGE_MEDIA_URL, document_root=settings.IMAGE_MEDIA_ROOT)
