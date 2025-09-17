# © 2025. Software anonimizado. Todos los derechos reservados.
# Consulte LICENSE para términos y condiciones.
import pickle
import os
import psutil  # Necesario para obtener la memoria RAM utilizada 
import sys
from .models import Dataset 
from .models import Actions
from .utils import UtilityHelper
from .models import Sample
__author__  = 'Anónimo' 
__title__   = 'Neurovectors Module'
__date__    = '2025-03-28' 
__version__ = '2.0.3'
__license__ = 'Proprietary' 


class O_Proyecto:
    def __init__(self) :
        """
        Objeto para guardar la configuración del proyecto
        """
        self.titulo     =   ''
        self.objetos        =   {'dataset':{},'engine':{},'collector':{},'analytics':{},'sample':{}}          # La lista de objetos de la pizarra
        self.version        =   __version__ # La versión del programa
        self.verbose        =   True        # Indica si escribe en la terminal todo lo que va haciendo
        self.seed           =   123         # La semilla por defecto que se usará si no se especifica una



class Proyecto:
    """
    Clase que gestiona el proyecto de manera global.

    Esta clase permite manejar el proyecto, incluyendo:

    - Carga y guardado de proyectos.
    - Gestión de datasets y engines.
    - Visualización de información general del proyecto.
    - Cálculo de memoria utilizada.

    Atributos:
        proyecto (O_Proyecto): Objeto que contiene la configuración del proyecto.
        pass_version (bool): Indica si hay que actualizar la versión del proyecto.
        file (str): Ruta al fichero del proyecto.
        actions (Actions): Objeto para gestionar acciones adicionales.

    Ejemplo de uso:

        >>> mi_proyecto = Proyecto(titulo='Análisis de Ventas', verbose=True)
        >>> dataset = mi_proyecto.add_dataset('ventas_2024')
        >>> dataset.file_path = '/ruta/al/archivo.csv'
        >>> mi_proyecto.show_project()
    """
    def __init__(self,file:str='',titulo:str='NUEVO',verbose:bool=True) :
        """
        Inicializa una instancia de la clase Proyecto.

        Args:
            file (str): Ruta al fichero del proyecto a cargar. Por defecto es una cadena vacía.
            titulo (str): Título del proyecto. Por defecto es 'NUEVO'.
            verbose (bool): Si es True, muestra información detallada en la terminal. Por defecto es True.
        """
        self.proyecto                = O_Proyecto()
        self.proyecto.titulo         =   titulo  # El nombre del proyecto
        self.proyecto.verbose        =   verbose # Indica si escribe en la terminal todo lo que va haciendo
        self.pass_version            =   True    # Indica si hay que actualizar la versión del proyecto
        self.file                    =   file    # El fichero si procede donde está el proyecto
        self.actions                 =   Actions(self.proyecto)
       
        
        if file!='':
            self.open_proyect(file=self.file)
        
        if self.proyecto.verbose:
            os.system('cls')
            print(f"Neurovectors Module {__version__}")
            print("----------------------------")
            print(f"Proyecto: {self.proyecto.titulo}")
            print("----------------------------")

    def open_proyect(self,file:str):
        """
        Abre un fichero que es el proyecto.

        Args:
            file (str): Ruta al fichero del proyecto.

        Raises:
            FileNotFoundError: Si el fichero no existe o no es válido.
        """
        self.file=file
        try:
            with open(file,"rb") as fichero:
                self.proyecto=pickle.load(fichero)
        except IOError as err:
            raise FileNotFoundError(f"The file '{file}' does not exist or is not a valid file.")     
        self.pass_version=False
        if self.proyecto.verbose:
            print("->Proyecto cargado...")

        if self.proyecto.version==__version__:
            self.pass_version=True
        else:
            self.proyecto.version=__version__
            if self.proyecto.verbose:
                print (f"Precaución! La versión cargada es {self.proyecto.version} y la del programa es {__version__}")
        
    def save_proyect(self,name_file:str=''):
        """
        Guarda el proyecto en un fichero.

        Args:
            name_file (str): Nombre del fichero donde se guardará el proyecto. Si es una cadena vacía, se usa el fichero actual.

        Raises:
            FileNotFoundError: Si ocurre un error al guardar el fichero.
        """
        if name_file=='':
            name_file=self.file
        for name,dataset in self.proyecto.objetos['dataset'].items():
            dataset.dataset_raw=None

        try:
            with open(name_file,"wb") as file:
                pickle.dump(self.proyecto,file)

        except IOError as err:
            raise FileNotFoundError(f"Error al guardar el archivo '{name_file}': {err}")
        
        if self.proyecto.verbose:
            print(f"Guardado: {name_file}")

    def add_dataset(self,name:str,file:str='',vars:list=[]):
        """
        Añade un dataset al proyecto.

        Args:
            name (str): Nombre que se asignará al dataset en el proyecto.

        Returns:
            Dataset: La instancia del dataset creado y añadido al proyecto.
        """
        self.proyecto.objetos['dataset'][name] = Dataset(self.proyecto.verbose,file=file,vars=vars,name=name)
        if self.proyecto.verbose:
            print (f"Dataset {name} añadido al proyecto")
            
        return self.proyecto.objetos['dataset'][name] 

    def drop_dataset(self,name:str):
        """
        Borra un dataset del proyecto
        """
        del self.proyecto.objetos['dataset'][name]
        if self.proyecto.verbose:
            print(f"Borrado el dataset {name}")
            
    def add_engine(self,name:str,type='EngineDict'):
        """
        Añade un motor al proyecto
        """
        match type:
            case 'EngineDict':
                from .models import Engine
                self.proyecto.objetos['engine'][name] = Engine(self.proyecto.verbose,name=name)

            case _:
                raise ValueError(f"El tipo de motor '{type}' no es válido.")
        
        if self.proyecto.verbose:
            print (f"Engine {name} de tipo {type} añadido al proyecto")
            
        return self.proyecto.objetos['engine'][name] 

    def drop_engine(self,name:str):
        """
        Borra un dataset del proyecto
        """
        del self.proyecto.objetos['engine'][name]
        if self.proyecto.verbose:
            print(f"Borrado el engine {name}")

    def add_analytics(self,name:str,type:str='Clas',method:str='Hits'):
        """
        Añade un motor al proyecto
        """
        match type:
            case 'Clas':
                from .models.analytics import Analytics
                self.proyecto.objetos['analytics'][name] = Analytics(name=name,verbose=self.proyecto.verbose,method=method)

            case _:
                raise ValueError(f"El tipo de análisis '{type}' no es válido.")
        
        if self.proyecto.verbose:
            print (f"Análisis {name} de tipo {type} añadido al proyecto")
            
        return self.proyecto.objetos['analytics'][name] 

    def drop_analytics(self,name:str):
        """
        Borra un dataset del proyecto
        """
        del self.proyecto.objetos['analytics'][name]
        if self.proyecto.verbose:
            print(f"Borrado el análisis {name}")
    
    def add_split(self,name_part1:str,name_part2:str,rows:int,percentage:float,seed:int=0,verbose:bool=True):
        """
        Añade una muestra de indices aleatorios al proyecto
        Devuelve dos tuplas con las filas correspondientes
        """
        if name_part1 in self.proyecto.objetos['sample'] or name_part2 in self.proyecto.objetos['sample']:
            raise ValueError("Los nombres de muestras ya existen") 
        
        if seed==0:
            seed=self.proyecto.seed

        part_1,part_2= UtilityHelper.ramdom_split(rows,percentage,seed)

        self.proyecto.objetos['sample'][name_part1]=Sample(verbose=verbose,name=name_part1,percentage=percentage,rows=part_1,complement=name_part2,seed=seed)
        self.proyecto.objetos['sample'][name_part2]=Sample(verbose=verbose,name=name_part2,percentage=(100-percentage),rows=part_2,complement=name_part1,seed=seed)
        return self.proyecto.objetos['sample'][name_part1],self.proyecto.objetos['sample'][name_part2]
    
    def get_split(self,name:str):
        """
        Devuelve una lista con los datos de la muestra
        """
        sample=self.proyecto.objetos['sample'][name]
        return sample
    
    def drop_split(self,name:str):
        """
        Borra una muestra del proyecto
        """
        del self.proyecto.objetos['sample'][name]
        if self.proyecto.verbose:
            print(f"Borradla muestra {name}")        

    def show_project(self):
        """
        Muestra la información del proyecto de manera global
        """
        # Obtener la memoria RAM que está ocupando el programa
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss  # En bytes
        ram_usage_fmt = self._sizeof_fmt(ram_usage)

        print("----------------------------------")
        print ("Neurovectors Server Edition")
        print(f"Proyecto: {self.proyecto.titulo}")
        print(f"Versión: {self.proyecto.version}")
        print(f"Memoria RAM utilizada por el programa: {ram_usage_fmt}")
        print("----------------------------")

        # Mostrar datasets
        print("Datasets:")
        total_dataset_size = 0
        if self.proyecto.objetos['dataset']:
            for name, dataset in self.proyecto.objetos['dataset'].items():
                size = self._get_size(dataset)
                total_dataset_size += size
                size_fmt = self._sizeof_fmt(size)
                print(f" - {name}:")
                dataset.show_dataset()
                print(f"    - Tamaño en memoria: {size_fmt}")
        else:
            print(" No hay datasets en el proyecto.")

        # Mostrar samples
        print("Samples:")
        total_samples_size = 0
        if self.proyecto.objetos['sample']:
            for name, sample in self.proyecto.objetos['sample'].items():
                size = self._get_size(sample)
                total_samples_size += size
                size_fmt = self._sizeof_fmt(size)
                sample.show_sample()
                print(f"    - Tamaño en memoria: {size_fmt}")
        else:
            print(" No hay muestras en el proyecto.")


        # Mostrar engines
        print("Engines:")
        total_engine_size = 0
        if self.proyecto.objetos['engine']:
            for name, engine in self.proyecto.objetos['engine'].items():
                size = self._get_size(engine)
                total_engine_size += size
                size_fmt = self._sizeof_fmt(size)
                engine.show_info()
                print(f"    - Tamaño en memoria: {size_fmt}")
        else:
            print(" No hay engines en el proyecto.")


        # Memoria total ocupada
        total_size = self._get_size(self.proyecto)
        print("----------------------------")
        print(f"Memoria ocupada por los datasets: {self._sizeof_fmt(total_dataset_size)}")
        print(f"Memoria ocupada por las muestras: {self._sizeof_fmt(total_samples_size)}")
        print(f"Memoria ocupada por los engines: {self._sizeof_fmt(total_engine_size)}")
        print(f"Memoria total ocupada por el proyecto: {self._sizeof_fmt(total_size)}")

    def _sizeof_fmt(self, num, suffix='B'):
        """Convierte tamaño en bytes a formato legible."""
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1024.0:
                return f"{num:3.1f} {unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f} Y{suffix}"
    
    def _get_size(self, obj, seen=None):
        """Recursivamente calcula el tamaño de un objeto en bytes."""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self._get_size(v, seen) for v in obj.values()])
            size += sum([self._get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += self._get_size(vars(obj), seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self._get_size(i, seen) for i in obj])
        return size
    



    # END
