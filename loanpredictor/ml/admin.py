from django.contrib import admin
from . import models


# Register your models here.

@ admin.register(models.Model)
class MlAdmin(admin.ModelAdmin):
    def time_seconds(self, obj):
        return obj.train_time.strftime("%H:%M:%S")
    time_seconds.admin_order_field = 'timefield'
    time_seconds.short_description = 'Training Time'    

    list_display = [ 'model_id', 'model_name','accuracy',
                    'recall','precision','time_seconds', 'created','active']
    
    readonly_fields = ['model_name','accuracy',
                    'recall','precision', 'created','time_seconds', 'train_samples',
                    'test_samples', 'train_time', 'max_depth','min_child_weight',
                    'learning_rate','subsample', 'colsample_bytree','colsample_bylevel',
                    'colsample_bynode', 'alpha','reg_lambda','gamma', 
                    'active_update_timestamp']
    
    list_filter = ['created',]
    search_fields = ['accuracy', 'recall']

    