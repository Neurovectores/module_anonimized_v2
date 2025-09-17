# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
# Anónimo

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm  # For progress bar
from ..utils import CSVHandler,DataDecomposer,UtilityHelper
from . import Sample

class Column:
    """
    Clase que define la información de las columnas del dataset
    """
    def __init__(self,type:str) -> None:
        self.type:str       = type
        self.exclude:bool   = False
        self.study:bool     = False
        self.log_difuse     = None
        self.numeric        = False
        self.is_numeric()

    def is_numeric(self):
        # Clasificar como tipo numérico o no
        if self.type.startswith('int'):
            self.type = 'integer'
            self.numeric = True
        elif self.type.startswith('float'):
            self.type = 'float'
            self.numeric = True
        else:
            self.type = 'str'
            self.numeric = False



class Dataset: 
    def __init__(self,verbose:bool=True,file:str='',vars:list=[],name:str='')->None:
        """
        Initializes the Dataset object.

        Args:
            verbose (bool): If True, additional information will be printed during processing.
        """
        self.file_path:str  = file                  # El fichero a trabajar
        self.verbose:bool   = verbose               # Si es true muestra en pantalla el log
        self.separator      = ','                   # El separador del csv
        self.header         = True                  # Indica si el csv tiene cabecera con los nombres de los campos
        self.chunk          = 100000                # El tamaño de trabajo del csv
        self.n_chunks       = 0                     # El número de chunks en que se ha partido el dataset
        self.dataset_raw    = pd.DataFrame()        # El dataframe de trabajo
        self.encoding       = 'latin-1'             # La codificación del csv
        self.columns_info:dict[str,Column]   = {}   # Diccionario con las columnas del csv
                                                    # col: {'exclude': False, 'study': False, 'log_difuse': None}
        self.n_rows         = 0                     # Número de filas del csv
        self.n_cols         = 0                     # Número de columnas del csv
        self.ready          = False                 # Indica que el dataset está operativo
        self.ready_train    = False                 # Indica si está listo para entrenar
        self.name           = name                  # El nombre del dataset
        if file!='':
            self.load_csv(file_path=file,chunk=self.chunk,encoding=self.encoding,separator=self.separator)
        
        if len(vars)!=0:
            self.set_study(vars)
    
    def load_csv(self,file_path:str,chunk:int=100000,encoding:str='latin-1',separator:str=','):
        """
        Loads the large CSV file into a data structure.
        If the file is detected to be in SVMlight format, it will be converted to CSV first.
        
        Args:
        file_path (str): Path to the CSV file.
        encoding (str): Encoding to be used when reading the CSV file. Default is 'latin-1'.
        chunk (int): Number of rows to read per chunk when processing large files.

        Raises:
        FileNotFoundError: If the file does not exist or is not a valid file.
        ValueError: If the chunk size is too small (< 5000).
        """

        # Step 1: Verify if the provided path exists and is a file 
        if self.verbose:
            print (f"->Star load dataset: {file_path}")
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            self.ready=False
            raise FileNotFoundError(f"The file '{file_path}' does not exist or is not a valid file.")
        if chunk<5000:
            self.ready=False
            raise ValueError(f"The number of rows to process is too small, use a value greater than 5000.")  
        self.file_path=file_path
        self.chunk=chunk
        self.encoding=encoding
        self.separator=separator

        # Step 2: Check if the file is in SVMlight format
        if CSVHandler.detect_svmlight_format(self.file_path):
            if self.verbose:
                print(f"The file '{self.file_path}' is in SVMlight format and must be converted to CSV before loading.")
            
            new_file_path = self.file_path + '.csv' 
            CSVHandler.svmlight_to_csv(self.file_path,new_file_path,self.verbose)
            self.file_path=new_file_path
                        # Comprobamos el separador
        separator = CSVHandler.detect_separator(self.file_path)
        if separator==None:
            if self.verbose:
                print(f"Could not determine the separator, using default: {self.separator}")

        elif separator!=self.separator:
            if self.verbose:
                print(f"The detected separator does not match the default, using new separator: {separator}")
            
            self.separator=separator

        self.header=CSVHandler.has_header(self.file_path)
        if self.header:
            self.dataset_raw = pd.read_csv(self.file_path, sep=self.separator,encoding=self.encoding,chunksize=self.chunk,on_bad_lines='skip',low_memory=False)
            if self.verbose:
                print("Header detected with field names.")
        else:
            self.dataset_raw = pd.read_csv(self.file_path, sep=self.separator,encoding=self.encoding,chunksize=self.chunk,on_bad_lines='skip',low_memory=False,header=None)
            if self.verbose:
                print("No header detected, using placeholder field names.")

        self.n_rows=0
        self.n_chunks=0

        # Step 4: Count rows and columns with a progress bar based on file size
        total_size = os.path.getsize(self.file_path)  # Get the total file size in bytes
        processed_size = 0  # To track the size of the data processed

        if self.verbose:
            print("Counting rows...")

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Processing CSV") if self.verbose else None as pbar:
            for chunk in self.dataset_raw:
                self.n_rows+=chunk.shape[0]
                self.n_chunks+=1
                # Update the processed size (bytes read)
                processed_size += chunk.memory_usage(deep=True).sum()
                # Update the progress bar based on the bytes processed
                if pbar:
                    pbar.update(processed_size - pbar.n)  # Update by the difference between the last and current processed size


        columnas= list(chunk.columns.values)
        # Crear un diccioanrio de columnas con su informaición
        self.columns_info = {}
        count_numeric=0
        for col in columnas:
            # Detectar tipo de datos basado en el primer chunk
            column_obj = Column(type=str(chunk[col].dtype))
            self.columns_info[col] = column_obj
            count_numeric+=column_obj.numeric

        self.n_cols=len(self.columns_info)
        if self.verbose:
            print(f"Found {self.n_rows} rows and {self.n_cols} columns ({count_numeric} are numerics) in the dataset.")
        self.ready=True

    def reset_df(self):
        """
        Recargar el csv para que se ponga el primer chunk del dataframe
        """
        if not self.ready:
            raise ValueError(f"The dataset is not ready")
         
        if self.header:
            self.dataset_raw = pd.read_csv(self.file_path, sep=self.separator,encoding=self.encoding,chunksize=self.chunk,on_bad_lines='skip',low_memory=False)
        else:
            self.dataset_raw = pd.read_csv(self.file_path, sep=self.separator,encoding=self.encoding,chunksize=self.chunk,on_bad_lines='skip',low_memory=False,header=None)
        
        self.ready_train = True
        self.ready       = True
        if self.verbose:
            print ("Dataset reloaded!")

    def set_study(self,columns:list):
        """
        Establece las variables de estudio
        """
        # quitamos todas las variables de estudio que había antes
        for properties in self.columns_info.values():
            properties.study = False
        
        self.ready_train=False
        for column in columns:
            if column in self.columns_info:
                self.columns_info[column].study=True
                self.ready_train=True
                if self.verbose:
                    print (f"Variable de estudio, {column} asignada")
            else:
                raise ValueError(f"La columna {column} no exite en el dataset")
            
    def set_exclude(self,columns:list):
        """
        Establece las columnas que vamos a exlcuir
        """
        # quitamos todas las variables de estudio que había antes
        for properties in self.columns_info.values():
            properties.exclude = False
        
        self.ready_train=False
        for column in columns:
            if column in self.columns_info:
                self.columns_info[column].exclude=True
                self.ready_train=True
                if self.verbose:
                    print (f"Columna, {column} excluida")
            else:
                raise ValueError(f"La columna {column} no exite en el dataset")

    # Método para obtener las columnas incluidas
    def get_included(self):
        """
        Devuelve una lista de nombres de columnas que no están excluidas.
        """
        return [col for col, info in self.columns_info.items() if not info.exclude]
    
    def get_vars(self):
        """
        Devuelve una lista de nombres de columnas que no están excluidas y que no son de estudio.
        """
        return [col for col, info in self.columns_info.items() if not info.exclude and not info.study]
     
    def get_study(self):
        """
        Obtiene la lista de las variables de estudio
        """
        # Obtenemos la variables de studio
        variable=[]
        for column, properties in self.columns_info.items():
            if properties.study:
                variable.append(column)            
                
        if len(variable)==0:
            self.ready_train=False
            raise ValueError(f"Don't study var select")
        else:
            self.ready_train=True
        
        return (variable)
    
    def export(self,filter_rows:Sample):
        """
        Divide el CSV original en dos archivos directamente en disco sin almacenar todo en memoria.
        '*_incluide.csv' contiene filas que están en self.aleatorio.
        '*_excluide.csv' contiene filas que no están en self.aleatorio.
        filter_rows: indica la lista de las filas que se van a filtrar
        """
        # Usar os.path.split para obtener la ruta del directorio y el nombre del archivo
        self.reset_df()
        ruta, archivo_con_extension = os.path.split(self.file_path)

        # Usar os.path.splitext para separar el nombre del archivo de su extensión
        nombre_archivo, extension = os.path.splitext(archivo_con_extension)

        # Generar nombres de archivo para las versiones 'include' y 'exclude'
        nombre_fichero_include = os.path.join(ruta, nombre_archivo + "_include" + extension)
        nombre_fichero_exclude = os.path.join(ruta, nombre_archivo + "_exclude" + extension)

        try:
            # Variable para controlar la escritura inicial con headers
            write_header = True
            # Procesar cada chunk con barra de progreso si verbose es True
            if self.verbose:
                # Iniciar la barra de progreso basada en el número de chunks
                chunks = tqdm(self.dataset_raw, total=self.n_chunks, desc="Export csv. Processing chunks")
            else:
                chunks = self.dataset_raw

            # Procesar cada chunk
            for chunk in chunks:
                chunk=UtilityHelper.convert_int(chunk)
                # Filtrar las filas basadas en self.aleatorio
                included_chunk = chunk[chunk.index.isin(filter_rows.rows)]
                excluded_chunk = chunk[~chunk.index.isin(filter_rows.rows)]
                # Escribir cada chunk filtrado a los archivos correspondientes
                mode = 'a' if not write_header else 'w'
                # Escribir cada chunk filtrado a los archivos correspondientes
                included_chunk.to_csv(nombre_fichero_include, mode=mode, header=write_header, index=False,sep=self.separator)
                excluded_chunk.to_csv(nombre_fichero_exclude, mode=mode, header=write_header, index=False,sep=self.separator)
                # Desactivar headers para las siguientes escrituras
                write_header = False

        except Exception as e:
            print(f"Error durante la carga y guardado de los CSVs: {e}")

    def to_digits(self,clean:float=0.05,delete:bool=True):
        """
        Descompone el csv cargado en sus dígitos menos la variable de estudio seleccionada
        añada la extensión _desc en el nombre del fichero 
        """
        if self.verbose:
            print (f"Descoponiendo el dataset en digitos...")
        # Usar os.path.split para obtener la ruta del directorio y el nombre del archivo
        self.reset_df()
        ruta, archivo_con_extension = os.path.split(self.file_path)
        # Usar os.path.splitext para separar el nombre del archivo de su extensión
        nombre_archivo, extension = os.path.splitext(archivo_con_extension)
        # Generar nombres de archivo para las versiones 'include' y 'exclude'
        nombre_fichero_desc = os.path.join(ruta, nombre_archivo + "_desc" + extension)
        handler_decomposer = DataDecomposer(columns_info=self.columns_info,tot_rows=self.n_rows,verbose=self.verbose)
        handler_decomposer.find_columns(df=self.dataset_raw,clean=clean,delete=delete)
        self.reset_df()
        handler_decomposer.decompose_dataframe(self.dataset_raw,nombre_fichero_desc,sep=self.separator)
        self.load_csv(nombre_fichero_desc,self.chunk,separator=self.separator)

    def process(self,action,title:str='Procesando dataset',row_start:int=0,rows:int=0,filter_rows:Sample=None, nan=None):
        """
        Función envolvente que procesa el dataset.
        """
        if not (self.ready and self.ready_train):
            raise ValueError(f"Dataset no preparado")
        
        self.reset_df()
        process_row=0
        
        # Comprobamos si nos han pasado una lista con las filas a filtrar
        if filter_rows!=None:
            # Ponemos el total de filas a filtrar
            randon_rows=filter_rows.n_rows
            mode=filter_rows.get_tittle()
        else:
            randon_rows=0
            mode='todas las filas'

        rows=UtilityHelper.check_limits(self.n_rows,row_start,rows,filter_max_rows=randon_rows)
        # Iniciamos el proceso del dataset
        with tqdm(total=rows, desc=title+F" ({mode})", unit="fila", disable=not self.verbose) as pbar:
            # Recorrer el dataset por chunks
            for chunk_raw in self.dataset_raw:
                # Filtrar columnas excluidas del chunk
                chunk = chunk_raw[self.get_included()]
                if randon_rows>0:
                    included_chunk = chunk[chunk.index.isin(filter_rows.rows)]
                else:
                    included_chunk = chunk

                if nan!=None:
                    included_chunk = included_chunk.fillna(nan)

                for index, row in included_chunk.iterrows():
                    process_row+=1
                    # Comprobación límites de la conversión
                    if process_row<row_start:
                        continue
                    if process_row>rows:
                        process_row-=1
                        if self.verbose:
                            pbar.close()
                            print (f"\nAlcanzado el límite de la conversión {process_row} filas")
                            
                        break
                    
                    # Llama a la función pasada por referencia
                    action(index,row)
                    # Actualizar la barra de progreso
                    pbar.update(1)
                    

        return process_row

    def show_dataset(self):
        print(f"    - Fichero asociado: {self.file_path or 'No especificado'}")
        print(f"    - Número de filas: {self.n_rows}")
        print(f"    - Número de columnas: {self.n_cols}")
        print(f"    - Listo para entrenar: {'Sí' if self.ready_train else 'No'}")
        print(f"    - En funcionamiento: {'Sí' if self.ready else 'No'}")

#END
