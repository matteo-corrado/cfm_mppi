"""Insert submodule root onto sys.path so cfm_mppi resolves to this submodule."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
