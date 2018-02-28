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

from . import calcs  # noqa
