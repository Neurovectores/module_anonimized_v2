# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo
from ..models.graph import Candidate,Cal_Method

class Nodes_Energy(Cal_Method):
    """
    Metodo que evalua por el número de impactos del neurovector en un alcance
    """
    def __init__(self) -> None:
        super().__init__()
        

    def evaluate_max(self,candidate:Candidate):
        if self.max is None or candidate.sum_nodes > self.max.sum_nodes:
            self.max = candidate