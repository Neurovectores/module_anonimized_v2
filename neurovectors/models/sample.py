# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
# Anónimo

import numpy as np
class Sample:
    def __init__(self,verbose:bool=True,name:str='',percentage:float=0,rows:np.ndarray=[],seed:int=0,complement:str='') :
        """
        Objeto de muestras aleatorias
        """
        self.name=name
        self.complement=complement
        self.percentage=percentage
        self.rows=rows
        self.seed=seed
        self.n_rows=len(rows)
        if verbose:
            self.show_sample()
    
    def show_sample(self):
        print(f"Muestra {self.name} complementario de {self.complement}")
        print(f"    --total filas: {self.n_rows}, porcentaje:{self.percentage} %, semilla {self.seed}")

    def get_tittle(self):
        return (f"{self.name} {self.percentage} %")
