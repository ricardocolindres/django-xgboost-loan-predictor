from django.db import models
from django.contrib.contenttypes.models import ContentType
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

# Create your models here.
class Model(models.Model):
    model_id = models.BigAutoField(primary_key=True)
    model_name = models.CharField(max_length=100, null=False, blank=False)
    model = models.BinaryField(null=False, blank=False)
    accuracy = models.DecimalField(decimal_places=2, max_digits=5, blank=True, null=True)
    recall = models.DecimalField(decimal_places=2, max_digits=5, blank=True, null=True)
    precision = models.DecimalField(decimal_places=2, max_digits=5, blank=True, null=True)
    train_samples = models.IntegerField(null=False, blank=False)
    test_samples = models.IntegerField(null=False, blank=False)
    train_time = models.TimeField(null=False, blank=False)
    max_depth = models.IntegerField(null=False, blank=False)
    min_child_weight = models.IntegerField(null=False, blank=False)
    learning_rate = models.FloatField(null=False, blank=False)
    subsample = models.FloatField(null=False, blank=False)
    colsample_bytree = models.FloatField(null=False, blank=False)
    colsample_bylevel = models.FloatField(null=False, blank=False)
    colsample_bynode = models.FloatField(null=False, blank=False)
    alpha = models.FloatField(null=False, blank=False)
    reg_lambda = models.FloatField(null=False, blank=False)
    gamma = models.FloatField(null=False, blank=False)
    created = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(null=False, blank=False, default=True)
    active_update_timestamp = models.DateTimeField(auto_now_add=False, auto_now=False, null=True, blank=True)

    def __str__(self):
        return f" Model #{self.model_id}"

    class Meta:
        ordering = ['-created']  
    
@receiver(pre_save, sender=Model, dispatch_uid="update_active_models")
def model_pre_save(sender, instance, *args, **kwargs):
        if instance.active:
            qs = sender.objects.all().exclude(model_id=instance.model_id)
            if qs.exists():
                qs.update(active=False, active_update_timestamp=timezone.now())


