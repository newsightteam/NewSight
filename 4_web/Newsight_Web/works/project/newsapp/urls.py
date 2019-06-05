from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from . import views

urlpatterns = [
    #path('admin/', admin.site.urls),
    # path('', views.index),
    # path('main_page/', views.page1),
    # path('cate_page/', views.page2),
    # path('topic_page/', views.page3),
    # path('news_page/', views.page4),
    #p ath('news_page/', views.post_list),
    path('main/', views.main),
    path('main/category/', views.query_result),
    path('main/category/topic/', views.category_result),
    path('main/category/topic/contents', views.detail_news_result),
    # url(r'\^?q=', views.post_list),
    # path('main_page/', views.test),
    path('test/', views.test)
]