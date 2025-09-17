# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo

import pandas as pd
from tqdm import tqdm
from .utility_helpers import UtilityHelper


class DataDecomposer:
    def __init__(self, columns_info:dict[str,'Column'],tot_rows:int=0, verbose:bool=True): # type: ignore
        """
        Initializes the DataDecomposer class with the variable to study.
        
        Parameters:
        study_var (str): Name of the variable to study that will not be decomposed.
        verbose (bool): Flag to indicate whether to print detailed messages.
        """
        from ..models.dataset import Column
        self.columns_info:dict[str,Column]  = columns_info   # type: ignore # La información con las columnas y como tratarlas
        self.columns = {}                   # El diccionario de columnas que se ha creado
        self.verbose = verbose              # Si Entre en modo de indicar lo que está haciendo
        self.tot_rows = tot_rows            # El total de filas a procesar, sirve para las barras de progreso
        self.exclude_columns:list   = []    # Lista con las columnas que se han excluido de la conversión
        self.template_df:pd.DataFrame = None # El template con la las columnas creadas
        # Generamos la lista de columnas que vamos a excluir
        for col,data in self.columns_info.items():
            if data.study or not data.numeric:
                self.exclude_columns.append('_'+col)

    def _breaknumber(self, column, value):
        """
        Decomposes a number into its digits and combines it with the column name.
        
        Parameters:
        column (str): The name of the column being decomposed.
        value (any): The value to be decomposed.
        
        Returns:
        dict: Dictionary containing the decomposed parts of the number with appropriate column names.
        """
        new_columns = {}
        is_number, value = UtilityHelper.isNumber(value)
        if not is_number:
            if value != "":
                new_columns[column] = value
            return new_columns  # Skip processing if the value is NaN or text
        
        value = str(value)
        if '.' in value:
            int_part, dec_part = value.split('.')
        else:
            int_part = value
            dec_part = '0'

        # Check the sign
        if '-' in int_part:
            int_part = int_part[1:]
            new_columns[f"{column}_neg"] = True

        # If there is a decimal part and the integer part is zero, ignore it
        if int(int_part) != 0 or int(dec_part) == 0:
            for index, digit in enumerate(reversed(int_part)):
                new_columns[f"{column}_a_{index}"] = digit

        if int(dec_part) > 0:
            for index, digit in enumerate(dec_part):
                new_columns[f"{column}_d_{index}"] = digit

        return new_columns
        
    def decompose_row(self, row):
        """
        Decomposes a row of a DataFrame and returns a dictionary of decomposed columns.
        
        Parameters:
        row (pd.Series): The row of the DataFrame to decompose.
        
        Returns:
        dict: Dictionary with the decomposed columns and their values.
        """
        decomposed = {}
        for col, value in row.items():
            
            if self.columns_info[col].study:
                decomposed['_'+col]=value
            else:
                if not self.columns_info[col].exclude:
                    decomposed.update(self._breaknumber(col, value))
        
        return decomposed
    
    def find_columns(self, df,clean=0.05,delete=True):
        """
        Iterates over the CSV file and creates a list of columns that form the decomposition.
        Also tracks the number of times each column is empty.
        
        Parameters:
        df (DataFrame or iterator): The DataFrame or an iterator of DataFrame chunks to analyze.
        tot_rows (int): Total number of rows in the DataFrame.
        clean (dupla): Dupla con los porcentajes de corte del uso de las variables para su limpieza
        
        Returns:
        DataFrame: An empty DataFrame with the structure of the found columns.
        """
        self.columns = {}
        progress_bar = tqdm(total=self.tot_rows, desc="Processing rows, find..", disable=not self.verbose)
        # First pass: determine all possible columns
        for chunk in df:
            for index, row in chunk.iterrows():
                decomposed_columns = self.decompose_row(row)
                for col in decomposed_columns.keys():
                    if col not in self.columns:
                        self.columns[col] =0  # Initialize empty count for new columns

                    self.columns[col]+= 1  # Track use column
                progress_bar.update(1)
        progress_bar.close()
        final_columns=[]
        tot_columns=len(self.columns)
        for col, use_count in self.columns.items():
            use_por=round(use_count/self.tot_rows,3)
            self.columns[col]=use_por
            study=False
            if col in self.columns_info:
                study=self.columns_info[col].study

            if not study and use_por <= clean:
                empty_por=100-(use_por*100)
                if delete:
                    if self.verbose:
                        print(f"Columna {col} omitida por {empty_por}% de vacios")
                else:
                    final_columns.append(col)
                    if self.verbose:
                        print(f"Recomendación de borrado de la columna {col} por tener {empty_por}% de vacios")
            else:
                final_columns.append(col)
        if self.verbose:
            print(f"Total columnas encontradas {tot_columns}, total columnas válidas {len(final_columns)}")

        sorted_columns = sorted(final_columns)      
        self.template_df= pd.DataFrame(columns=sorted_columns)

    def decompose_dataframe(self, df, output_file, sep=';'):
        """
        Decomposes an input DataFrame into its digit-based representation using the template DataFrame and writes it to a file.
        Parameters:
        df (DataFrame or iterator): The DataFrame or an iterator of DataFrame chunks to decompose.
        tot_rows (int): Total number of rows in the DataFrame.
        output_file (str): Path to the output CSV file.
        sep (str): Separator to use in the CSV file. Default is ';'.
        """
        header_written = True
        mode='w'
        progress_bar = tqdm(total=self.tot_rows, desc="Processing rows, decompose..", disable=not self.verbose)
        for chunk in df:
            decomposed_data = self.template_df.copy()
            for index, row in chunk.iterrows():
                decomposed_row = self.decompose_row(row)
                for col, value in decomposed_row.items():
                    if col in decomposed_data.columns:  # Verificar si la columna existe antes de agregar el valor
                        decomposed_data.at[index, col] = value
            
            # Write the chunk to the CSV file
            decomposed_data=UtilityHelper.convert_int(df=decomposed_data,info_columns=self.exclude_columns)
            decomposed_data.to_csv(output_file, sep=sep, mode=mode, header=header_written, index=False, encoding='utf-8')
            if header_written:
                header_written=False
                mode='a'
            
            progress_bar.update(len(chunk))
        progress_bar.close()
