# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from pathlib import Path

# -------------------------------------------------------------------------------------
# ARKIMEDE ROOT
# -------------------------------------------------------------------------------------

def arkimede_root():
    """Return the root directory of the installed arkimede package."""
    import arkimede
    return Path(arkimede.__file__).parent.parent

# -------------------------------------------------------------------------------------
# CHECKPOINTS BASEDIR
# -------------------------------------------------------------------------------------

def checkpoints_basedir():
    """Return the root directory of the ocp checkpoints path."""
    return arkimede_root() / "checkpoints"

# -------------------------------------------------------------------------------------
# TEMPLATES BASEDIR
# -------------------------------------------------------------------------------------

def templates_basedir():
    """Return the root directory of the arkimede templates path."""
    return arkimede_root() / "templates"

# -------------------------------------------------------------------------------------
# SCRIPTS BASEDIR
# -------------------------------------------------------------------------------------

def scripts_basedir():
    """Return the root directory of the arkimede scripts path."""
    return arkimede_root() / "scripts"

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------