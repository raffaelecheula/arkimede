# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from ocpmodels.common.relaxation.ase_utils import OCPCalculator

# -----------------------------------------------------------------------------
# OCP CALCULATOR COUNTER
# -----------------------------------------------------------------------------

class OCPCalculatorCounter(OCPCalculator):
        
    def __init__(
        self,
        config_yml=None,
        checkpoint_path=None,
        trainer=None,
        cutoff=6,
        max_neighbors=50,
        cpu=True,
    ):
        super().__init__(
            config_yml=config_yml,
            checkpoint_path=checkpoint_path,
            trainer=trainer,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            cpu=cpu,
        )
        self.n_singlepoints = 0
    
    def calculate(self, atoms, properties, system_changes):
        self.n_singlepoints += 1
        super().calculate(atoms, properties, system_changes)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------