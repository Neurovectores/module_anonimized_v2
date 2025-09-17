# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
# Anónimo

import pandas as pd
import random
import numpy as np
from babel.numbers import parse_decimal

class UtilityHelper:
    @staticmethod
    def convert_int(df: pd.DataFrame, info_columns: list):
        """
        Convierte las columnas del DataFrame a enteros cuando sea posible,
        manejando valores NaN, excluyendo las columnas especificadas.

        Args:
            df (pd.DataFrame): DataFrame a procesar.
            exclude_columns (list, opcional): Lista de nombres de columnas a excluir de la conversión.
        
        Returns:
            pd.DataFrame: DataFrame con las columnas convertidas a enteros cuando sea posible.
        """
        for col in df.columns:
            # Saltar las columnas que están en la lista de exclusión
            if col in info_columns:
                continue
            
            try:
                # Intentar convertir a numérico (para manejar NaN y posibles números decimales)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar si todos los valores no NaN son equivalentes a enteros
                non_nan_values = df[col].dropna()
                if (non_nan_values % 1 == 0).all():
                    df[col] = df[col].astype('Int64')  # Convertir a int, manejando NaN con Int64
            except (ValueError, TypeError):
                pass  # Si no se puede convertir, dejar la columna como está
        
        return df

    
    @staticmethod
    def isNumber(value:str):
        """
        Checks if the value is numeric and returns it in a formatted way.
        
        Parameters:
        value (any): The value to check.
        
        Returns:
        tuple: A tuple containing a boolean indicating if the value is numeric and the formatted value.
        """
        
        if value != value or value==None:  # Check for NaN or None
            value = ""
            return False, value
        value=str(value)
        try:
            float(value)
            value = parse_decimal(str(value), locale='en')
            return True, value
        except ValueError:
            return False, value
        
    @staticmethod
    def check_limits(total_rows:int,row_start:int,rows:int,filter_max_rows:int=0)->int:
        """
        Comprueba el número de filas a procesar. Lo cambia si es menor que 0
        """
        if rows<=0:
            if filter_max_rows>0:
                rows=filter_max_rows
            else:
                rows=total_rows-row_start
        else:
            if filter_max_rows>0 and rows>filter_max_rows:
                rows=filter_max_rows

        if row_start>=total_rows or row_start+rows>total_rows:
            raise ValueError(f"Error en los límetes de la conversión")
        
        return rows
    
    @staticmethod
    def ramdom_split(rows, percentage,seed=123):
        """
        Generate a sample of random row indices based on the input and return the remaining indices.
        
        Parameters:
        rows (int or list): Number of rows or list of row indices.
        percentage (float): Percentage of rows to select.
        seed (int, optional): Seed for the random number generator.
        
        Returns:
        tuple: A tuple containing two NumPy arrays - the selected sample indices and the remaining indices.
        """
        random.seed(seed)
        if isinstance(rows, int):
            # Case when rows is an integer representing the total number of rows
            all_indices = list(range(rows))
            num_rows = int(rows * (percentage / 100))
            selected_indices = random.sample(all_indices, num_rows)
        elif isinstance(rows, np.ndarray):
            # Case when rows is a list of row indices
            all_indices = rows
            num_rows = int(len(rows) * (percentage / 100))
            selected_indices = random.sample(list(all_indices), num_rows)
        else:
            print (rows)
            raise ValueError("Input 'rows' must be an integer or a list of row indices.")

        remaining_indices = list(set(all_indices) - set(selected_indices))
        
        return np.array(selected_indices), np.array(remaining_indices)
    
    @staticmethod
    def isEmpty(value:any)->bool:
        """
        Verifica si un valor lo consideramos que está vacio
        """
        return (value!=value or value=='' or value==None)

    #END

    
