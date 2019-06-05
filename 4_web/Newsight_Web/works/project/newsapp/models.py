# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

class TotalDb(models.Model):
    id = models.TextField(primary_key=True)
    date = models.TextField(blank=True, null=True)
    title = models.TextField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    category = models.TextField(blank=True, null=True)
    date_tmp = models.BigIntegerField(blank=True, null=True)
    tokenized_doc = models.TextField(blank=True, null=True)
    tokenized_title = models.TextField(blank=True, null=True)
    query = models.TextField(blank=True, null=True)


    class Meta:
        managed = False
        db_table = 'total_db'
