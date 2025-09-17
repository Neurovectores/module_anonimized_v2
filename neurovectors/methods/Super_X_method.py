# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Nuevo método de cálculo "Super_X"

from ..models.graph import Candidate, Cal_Method
import random

class Super_X(Cal_Method):
    """
    Método que combina estrategias de ensemble, bootstrap y priorización de nodos
    para mejorar la precisión en niveles bajos de certeza.
    """
    def __init__(self, certainty_threshold=0.4, top_n_neurovectors=10, bootstrap_size=5):
        """
        Inicializa el método Super_X con parámetros de ensemble y bootstrap.

        Args:
            certainty_threshold (float): Umbral de certeza para activar expansión.
            top_n_neurovectors (int): Número de neurovectores de mayor energía a considerar.
            bootstrap_size (int): Tamaño del subconjunto aleatorio de neurovectores.
        """
        super().__init__()
        self.certainty_threshold = certainty_threshold
        self.top_n_neurovectors = top_n_neurovectors
        self.bootstrap_size = bootstrap_size

    def expand_nodes_with_energy(self, candidate, neurovectors):
        """
        Expande el conjunto de nodos coincidentes usando los neurovectores con mayor energía.

        Args:
            candidate (Candidate): El candidato a evaluar.
            neurovectors (list): Lista de neurovectores disponibles.
        """
        # Seleccionamos los top_n neurovectores con mayor energía
        top_nv = sorted(neurovectors, key=lambda nv: nv.energy, reverse=True)[:self.top_n_neurovectors]

        for nv in top_nv:
            # Agregar nodos del neurovector al conjunto de nodos coincidentes
            for node in nv.nodes:
                if node not in candidate.node_hits:
                    candidate.node_hits.add(node)
                    candidate.certainty_rel += 0.01  # Incremento marginal en certeza

    def bootstrap_neurovectors(self, neurovectors):
        """
        Selecciona un subconjunto aleatorio de neurovectores para diversificar la evaluación.

        Args:
            neurovectors (list): Lista de neurovectores disponibles.

        Returns:
            list: Subconjunto aleatorio de neurovectores.
        """
        if isinstance(neurovectors, list):
            return random.sample(neurovectors, min(self.bootstrap_size, len(neurovectors)))
        else:
            return [neurovectors]  # Si es un único neurovector, lo devolvemos como lista

    def ensemble_predictions(self, candidates, methods):
        """
        Combina predicciones de múltiples métodos mediante un esquema de ensemble.

        Args:
            candidates (list): Lista de candidatos a evaluar.
            methods (list): Lista de métodos alternativos para evaluar.

        Returns:
            Candidate: El candidato con el mayor puntaje combinado.
        """
        scores = {candidate: 0 for candidate in candidates}
        for method in methods:
            for candidate, score in method.evaluate_candidates(candidates).items():
                scores[candidate] += score
        return max(scores, key=scores.get)  # Candidato con mayor puntaje

    def evaluate_max(self, candidate: Candidate):
        """
        Evalúa el candidato aplicando estrategias de ensemble, bootstrap y expansión de nodos.

        Args:
            candidate (Candidate): El candidato a evaluar.
        """
        # Si la certeza es baja, aplicamos expansión y bootstrap
        if candidate.certainty_rel < self.certainty_threshold:
            neurovectors = self.bootstrap_neurovectors(candidate.neurovector)
            self.expand_nodes_with_energy(candidate, neurovectors)

        # Evaluar el candidato basado en el conjunto expandido de nodos
        candidate.score = sum(node.energy for node in candidate.node_hits)

        if self.max is None or candidate.score > self.max.score:
            self.max = candidate

    def reset(self, candidate: Candidate, scope: int):
        """
        Reinicia el método con un nuevo candidato inicial.

        Args:
            candidate (Candidate): El candidato inicial.
            scope (int): Alcance del cálculo.
        """
        super().reset(candidate, scope)
        candidate.score = sum(node.energy for node in candidate.node_hits)
        self.max = candidate

    def get_mae(self):
        """
        Calcula el MAE del candidato seleccionado.

        Returns:
            tuple: Indica si es numérico, el MAE y la predicción media.
        """
        numeric = self.max.var.numeric
        mae = 0
        predicted_value = None

        if numeric:
            predicted_value = self.max.predict.value if self.max.predict else None
            mae = abs(self.max.var.value - predicted_value) if predicted_value else abs(self.max.var.value)

        return numeric, mae, predicted_value

    def get_success(self):
        """
        Determina si la predicción es exitosa.

        Returns:
            bool: True si la predicción es exitosa, False en caso contrario.
        """
        return self.max.success
