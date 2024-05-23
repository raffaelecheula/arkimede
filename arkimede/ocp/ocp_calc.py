# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ocpmodels.common.relaxation.ase_utils import OCPCalculator

# -------------------------------------------------------------------------------------
# CALCULATOR COUNTER
# -------------------------------------------------------------------------------------

class CalculatorCounter(object):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0
    
    def calculate(self, **kwargs):
        super().calculate(**kwargs)
        self.counter += 1

# -------------------------------------------------------------------------------------
# OCP CALCULATOR COUNTER
# -------------------------------------------------------------------------------------

class OCPCalculatorCounter(OCPCalculator):
        
    def __init__(self, adjust_sum_force=False, **kwargs):
        super().__init__(**kwargs)
        self.adjust_sum_force = adjust_sum_force
        self.counter = 0
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        if self.adjust_sum_force is True:
            self.results["forces"] -= self.results["forces"].mean(axis=0)
        self.counter += 1

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------