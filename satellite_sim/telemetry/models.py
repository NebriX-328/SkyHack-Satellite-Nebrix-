from django.db import models

class TelemetryUpload(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class TelemetryData(models.Model):
    timestamp = models.DateTimeField()
    temperature = models.FloatField(null=True, blank=True)
    battery = models.FloatField(null=True, blank=True)
    solar = models.FloatField(null=True, blank=True)
    radiation = models.FloatField(null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    speed = models.FloatField(null=True, blank=True)
    orientation = models.FloatField(null=True, blank=True)
    anomaly = models.CharField(max_length=255, blank=True)
