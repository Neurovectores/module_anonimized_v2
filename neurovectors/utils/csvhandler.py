# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo

import pandas as pd
import csv
import os
import tempfile
from sklearn.datasets import load_svmlight_file

class CSVHandler:
    """
    Clase auxiliar para manejar archivos CSV y formatos similares,
    incluyendo detección de encabezados, separadores y conversión
    desde formato SVMlight a CSV.
    """

    @staticmethod
    def has_header(file_path, num_lines=4):
        """
        Verifica si un archivo CSV tiene encabezado.

        Args:
            file_path (str): Ruta al archivo CSV.
            num_lines (int): Número de líneas a leer para detectar el encabezado.

        Returns:
            bool: True si el archivo tiene encabezado, False en caso contrario.
        """
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                sample = ''.join([file.readline() for _ in range(num_lines)])
                sniffer = csv.Sniffer()
                return sniffer.has_header(sample)
        except (csv.Error, Exception):
            return False

    @staticmethod
    def detect_separator(file_path, num_lines=3):
        """
        Detecta el separador utilizado en un archivo CSV.

        Args:
            file_path (str): Ruta al archivo CSV.
            num_lines (int): Número de líneas a leer para detectar el separador.

        Returns:
            str: El separador detectado, o None si no se pudo detectar.
        """
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                sample = ''.join([file.readline() for _ in range(num_lines)])
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                return dialect.delimiter
        except (csv.Error, Exception):
            return None

    @staticmethod
    def detect_svmlight_format(file_path, max_lines=10):
        """
        Verifica si un archivo está en formato SVMlight.

        Args:
            file_path (str): Ruta al archivo a verificar.
            max_lines (int): Número máximo de líneas a analizar.

        Returns:
            bool: True si el archivo está en formato SVMlight, False en caso contrario.
        """
        try:
            # Leer un número limitado de líneas del archivo original
            with open(file_path, 'r') as file:
                lines = [next(file) for _ in range(max_lines)]

            # Escribir esas líneas en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                temp_file.writelines(lines)
                temp_file_path = temp_file.name

            # Intentar cargar el archivo temporal usando scikit-learn
            load_svmlight_file(temp_file_path)
            os.remove(temp_file_path)  # Eliminar el archivo temporal
            return True
        except Exception:
            return False

    @staticmethod
    def svmlight_to_csv(svmlight_path, csv_path):
        """
        Convierte un archivo SVMlight a CSV.

        Args:
            svmlight_path (str): Ruta al archivo SVMlight.
            csv_path (str): Ruta donde se guardará el archivo CSV.
        """
        try:
            X, y = load_svmlight_file(svmlight_path)

            if X is None or y is None:
                print("No se pudieron cargar los datos.")
                return

            # Convertir a DataFrame de pandas
            df_X = pd.DataFrame(X.toarray())  # Convertir matriz dispersa a densa
            df_y = pd.Series(y, name='label')

            # Concatenar las características y las etiquetas
            df = pd.concat([df_y, df_X], axis=1)

            # Guardar como CSV
            df.to_csv(csv_path, index=False)
            print(f'Archivo CSV guardado en: {csv_path}')
        except Exception as e:
            print(f'Error al convertir SVMlight a CSV: {e}')
