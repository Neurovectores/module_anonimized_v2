# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo
from ..models.graph import Candidate,Cal_Method

class Hits(Cal_Method):
    """
    Metodo que evalua por el número de impactos del neurovector en un alcance
    """
    def __init__(self) -> None:
        super().__init__()
        
    def evaluate_max(self,candidate:Candidate):
        eval_hits=candidate.tot_hits-self.max.tot_hits
        eval_energy=(candidate.neurovector.energy>self.max.neurovector.energy)
        if eval_hits>0 or (eval_hits==0 and eval_energy):
            self.max = candidate
