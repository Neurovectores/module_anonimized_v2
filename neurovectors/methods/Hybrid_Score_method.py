# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Nuevo método de cálculo "Hybrid_Score"

from ..models.graph import Candidate, Cal_Method
import numpy as np

class Hybrid_Score(Cal_Method):
    """
    Método que evalúa la predicción basada en el promedio ponderado de los valores predichos,
    ajustando la ponderación según la certeza de los candidatos.
    """
    def __init__(self, certainty_threshold=0.75):
        """
        Inicializa el método Hybrid_Score con ajustes dinámicos basados en certeza.

        Args:
            certainty_threshold (float): Umbral de certeza para ajustar la ponderación.
        """
        super().__init__()
        self.acc_value = 0  # Acumulador de valores predichos ponderados
        self.acc_weight = 0  # Acumulador de pesos
        self.avg_predict = None  # Predicción media
        self.certainty_threshold = certainty_threshold

    def evaluate_max(self, candidate):
        """
        Evalúa el candidato acumulando valores ponderados basados en certeza.

        Args:
            candidate (Candidate): El candidato a evaluar.
        """
        if candidate.predict is not None and candidate.predict.numeric:
            value = candidate.predict.value
            weight = candidate.tot_hits

            # Ajustar el peso basado en la certeza
            if candidate.certainty_rel >= self.certainty_threshold:
                weight *= 1.5  # Incrementamos el peso en niveles altos de certeza
            else:
                weight *= 0.8  # Reducimos el peso en niveles bajos de certeza

            self.acc_value += value * weight
            self.acc_weight += weight

    def reset(self, candidate, scope):
        """
        Reinicia acumuladores y calcula una predicción inicial basada en el candidato.

        Args:
            candidate (Candidate): El candidato inicial.
            scope (int): Alcance del cálculo.
        """
        super().reset(candidate, scope)
        self.acc_value = 0
        self.acc_weight = 0
        self.avg_predict = None
        if candidate.predict is not None and candidate.predict.numeric:
            weight = candidate.tot_hits * scope * 1.1
            if candidate.certainty_rel >= self.certainty_threshold:
                weight *= 1.5
            else:
                weight *= 0.8

            self.acc_value = candidate.predict.value * weight
            self.acc_weight = weight
            if self.acc_weight > 0:
                self.avg_predict = self.acc_value / self.acc_weight

    def get_mae(self):
        """
        Calcula el MAE basado en la predicción media.

        Returns:
            tuple: Indica si es numérico, el MAE y la predicción media.
        """
        numeric = self.max.var.numeric
        mae = 0
        if self.acc_weight > 0:
            self.avg_predict = self.acc_value / self.acc_weight
        if numeric and self.avg_predict is not None:
            mae = abs(self.avg_predict - self.max.var.value)
        return numeric, mae, self.avg_predict

    def get_success(self):
        """
        Determina si la predicción es exitosa comparando la predicción media redondeada.

        Returns:
            bool: True si la predicción es exitosa, False en caso contrario.
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
