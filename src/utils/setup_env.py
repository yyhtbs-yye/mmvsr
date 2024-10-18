# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import importlib
import warnings
from types import ModuleType
from typing import Optional

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmvsr into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmvsr default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmvsr`, and all registries will build modules from
            mmvsr's registry node.
            To understand more about the registry, please refer
            to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html
            Defaults to True.
    """  # noqa
    import src.datasets  # noqa: F401,F403
    import src.engine  # noqa: F401,F403
    import src.evaluation  # noqa: F401,F403
    import src.models  # noqa: F401,F403
    import src.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
            or not DefaultScope.check_instance_created('mmvsr')
        if never_created:
            DefaultScope.get_instance('mmvsr', scope_name='mmvsr')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmvsr':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmvsr", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmvsr". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmvsr-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmvsr')


def try_import(name: str) -> Optional[ModuleType]:
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
