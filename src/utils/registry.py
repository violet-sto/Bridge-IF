from src.datamodules import DATAMODULE_REGISTRY
from src.models import MODEL_REGISTRY
from src.tasks import TASK_REGISTRY

registry_dict = dict(
    datamodule=DATAMODULE_REGISTRY,
    task=TASK_REGISTRY,
    model=MODEL_REGISTRY
)

def get_module(group_name, module_name):
    group = registry_dict.get(group_name, None)
    if group is None:
        raise KeyError(f'{group_name} is not a valid registry group {registry_dict.keys()}.')
    
    return group.get(module_name)

def get_registered_modules(group_name):
    group = registry_dict.get(group_name)
    if group is not None:
        return group.keys()
    else:
        raise KeyError(f'{group_name} is not a valid registry group {registry_dict.keys()}.')

__all__ = [
    'get_module',
    'get_registered_modules'
]