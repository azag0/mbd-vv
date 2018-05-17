from pkg_resources import resource_string

from caf import Caf, Cellar
from caf.executors import DirBashExecutor, DirPythonExecutor
from caf.Tools.aims import AimsTask

kcal = 627.503
ev = 27.2113845

app = Caf()
cellar = Cellar(app)
dir_bash = DirBashExecutor(app, cellar)
dir_python = DirPythonExecutor(app, cellar)
aims = AimsTask(dir_bash)


def use_old_sr_basis(task):
    key = task['speciedir'], 'Sr'
    if key not in aims.basis_defs:
        aims.basis_defs[key] = resource_string(__name__, 'data/38_Sr_default').decode()


aims.features.insert(1, use_old_sr_basis)

from . import calcs  # noqa
