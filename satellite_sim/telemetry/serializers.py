from rest_framework import serializers
from .models import TelemetryData

class TelemetryDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TelemetryData
        fields = '__all__'
