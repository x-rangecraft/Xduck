"""Register the external task and Hydra owner for tests."""

import os

os.environ.setdefault("UNILAB_EXTRA_REGISTRY_PACKAGES", "microduck_rl_unilab.tasks")

from microduck_rl_unilab.conf_searchpath import register_conf_search_path

register_conf_search_path()
