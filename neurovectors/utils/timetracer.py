# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo

import time

class TimeTracer:
    """
    Clase auxiliar para medir y registrar los tiempos de ejecución en diferentes partes del código.
    """

    def __init__(self):
        self.logs = {}             # Registro de mediciones temporales
        self.is_ready = True       # Estado del temporizador; True si está parado
        self.total_time = 0        # Tiempo total transcurrido
        self.previous_label = ''   # Etiqueta del punto anterior
        self.counter = 0           # Número total de puntos registrados
        self.start_time = 0        # Tiempo de inicio del temporizador

    def start(self, label=''):
        """
        Inicia o registra un nuevo punto en el temporizador.

        Args:
            label (str): Etiqueta para el punto de tiempo.
        """
        current_time = time.time()
        if not label:
            label = f"Point {self.counter}"

        if self.is_ready:
            self.logs = {label: {'time': current_time, 'elapsed': 0, 'previous': ''}}
            self.previous_label = label
            self.is_ready = False
            elapsed = 0
            self.counter += 1
            self.start_time = current_time
        else:
            elapsed = current_time - self.logs[self.previous_label]['time']
            self.logs[label] = {'time': current_time, 'elapsed': elapsed, 'previous': self.previous_label}
            self.previous_label = label
            self.counter += 1

        return elapsed

    def stop(self, label=''):
        """
        Detiene el temporizador y devuelve el tiempo total transcurrido.

        Args:
            label (str): Etiqueta para el punto final.

        Returns:
            float: Tiempo total transcurrido.
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        final_label = label if label else f"End ({self.counter})"
        self.logs[final_label] = {'time': current_time, 'elapsed': elapsed, 'previous': self.previous_label}
        self.is_ready = True
        self.total_time = round(elapsed, 4)
        return self.total_time

    def show(self):
        """
        Muestra el registro de tiempos.
        """
        for label, data in self.logs.items():
            print(f"{label}: {round(data['elapsed'], 6)} segundos")
        print("------------------------------------")
        print(f"TIEMPO TOTAL: {self.total_time} segundos")

    def get_total_time(self):
        """
        Obtiene el tiempo total transcurrido.

        Returns:
            float: Tiempo total transcurrido.
        """
        return self.total_time
