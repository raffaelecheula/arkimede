# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml

# -------------------------------------------------------------------------------------
# READ STEP ACTLEARN
# -------------------------------------------------------------------------------------

def read_step_actlearn(filename):
    """Get the current step of the active learning loop."""
    if os.path.isfile(filename):
        with open(filename, "r") as fileobj:
            step = yaml.safe_load(fileobj)["step_actlearn"]
    else:
        step = 0
        write_step_actlearn(filename=filename, step=step)
    return step

# -------------------------------------------------------------------------------------
# WRITE STEP ACTLEARN
# -------------------------------------------------------------------------------------

def write_step_actlearn(filename, step):
    """Write the new step of the active learning loop."""
    with open(filename, "w") as fileobj:
        yaml.dump({"step_actlearn": step}, fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
