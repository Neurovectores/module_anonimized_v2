# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo
from ..models.graph import Candidate,Cal_Method

class Avg(Cal_Method):
    """
    Metodo que evalua por la media de los resultados.
    Coge el primer cadidato como resultado para almacenar los datos
    """
    def __init__(self) -> None:
        super().__init__()
        # Acumuladores temporales para el cálculo de la predicción média
        self.acc_value                  = 0    # Acumulador de lo valores predichos X el KPI usado para hacer la predicción de regresión
        self.acc_weight                 = 0    # Acumulador del ponderador usado para hacer la predicción de regresión
        self.avg_predict                = None # La predicción média final


    def evaluate_max(self,candidate:Candidate):        
        # Acumulamos los datos para hacer el cálculo de predicción média
        if candidate.predict is not None and candidate.predict.numeric:
            value=candidate.predict.value
            #weight=candidate.tot_hits
            #weight=candidate.sum_nodes
            weight=1
            self.acc_value  += value*weight
            self.acc_weight += weight

    def reset(self,candidate:Candidate,scope:int)->None:
        """
        Implementación del método reset
        """
        super().reset(candidate=candidate)
        self.avg_predict = None
        self.acc_value=0    # Acumulador de lo valores predichos X el KPI usado para hacer la predicción de regresión
        self.acc_weight=0   # Acumulador del ponderador usado para hacer la predicción de regresión
        if candidate.predict is not None and candidate.predict.numeric:
            #weight=candidate.tot_hits
            #weight=candidate.sum_nodes
            weight=1
            self.acc_value                  = candidate.predict.value*weight  
            self.acc_weight                 = weight    
            if self.acc_weight>0:
                self.avg_predict = self.acc_value/self.acc_weight


    def get_mae(self)->tuple[bool,float]:
        """
        Optiene el MAE específico de la media de los resultados
        """
        numeric = self.max.var.numeric
        mae     = 0
        self.avg_predict=None
        if self.acc_weight>0:
            self.avg_predict = self.acc_value/self.acc_weight

        if numeric:
            # Calculamos el valor medio de la predicción y lo almacenamos en la predicción
            if self.avg_predict!=None:
                mae=abs(self.avg_predict-self.max.var.value)
            else:
                mae=abs(self.max.var.value)

        return numeric, mae, self.avg_predict
    

    def get_success(self)->bool:
        """
        Resuelve si se ha tenido éxito o no, En este cálculo es distinto al resto
        """
        success=None
        if self.max.var.numeric:
            if self.avg_predict!=None:
                avg_predict=round(self.avg_predict,0)
                if avg_predict==self.max.var.value:
                    success=True
                else:
                    success=False
            else:
                success=False
        
        return success