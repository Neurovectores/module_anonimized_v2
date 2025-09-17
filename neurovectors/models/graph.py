# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
#
# This software is provided solely for non-commercial use.  
# Refer to the LICENSE file for full terms and conditions. 

from ..utils.utility_helpers import UtilityHelper 
import pandas as pd
import math
from collections import Counter
from abc import ABC, abstractmethod

class Node:
    """
    Clase principal que define el objeto Nodo, dentro está el índice principal
    de propagación en los neurovectores
    """
    def __init__(self,id:int,column:str,value:str,name:str) -> None:
        # KPIS del nodo
        self.acc_use:int        = 0             # Veces usado el nodo
        self.acc_use_var:int    = 0             # Veces que el nodo se usa como salida
        self.acc_successes:int  = 0             # Veces del éxito del nodo
        self.acc_residue:int    = 0             # Veces que ha llegado la energía residual
        self.acc_MSE:float      = 0             # El MSE acumulado
        self.acc_MAE:float      = 0             # El MAE acumulado
        self.acc_MAE_use        = 0             # El uso de este neurovector cuando se usa como regresión
        # KPIS calculados
        self.energy:float       = 0             # La energia del nodo según el cálculo
        self.RMSE:float          = 0            # El MSE del nodo
        self.MAE:float          = 0             # El MAE del nodo
        # Variables de gobierno
        self.column:str         = column        # La columna asociada al nodo
        self.value              = value         # El valor de la celda
        self.numeric,_          = UtilityHelper.isNumber(value)            # Si es numérico
        self.id:int             = id            # El id del nodo
        self.name:str           = name          # El literal del nodo
        self.nvs:set            = set()         # El listado de neurovectores donde está relacionado

    
    def cal(self,mae:float,mse:float,success:bool,is_var:bool=False):
        """
        El cálculo de los KPI calculados
        """
        self.acc_use+=1
        self.acc_use_var+=is_var
        self.acc_successes+=success
        self.energy=self.acc_successes/self.acc_use
        if mae!=None:
            self.acc_MAE_use+=1
            self.acc_MAE+=mae
            self.acc_MSE+=mse
            self.MAE=self.acc_MAE/self.acc_use
            self.RMSE=math.sqrt(self.acc_MSE/self.acc_use)
      
    def show(self):
        print (f"   -- Node ({self.id}): {self.column} {self.value} (Number: {self.numeric}), Use: {self.acc_use}, Successes: {self.acc_successes}, Energía: {self.energy},  Residue: {self.acc_residue}, MAE: {self.MAE}, MSE: {self.MSE}")

class Neurovector:
    """
    Clase que define el objeto Neurovector
    """
    def __init__(self,row:int,id:int,epoch:int) -> None:
        # KPIS del Neurovector
        self.acc_use        = 0                 # Veces que se ha usado para hacer una predicción
        self.acc_successes  = 0                 # Veces que ha tenido éxito
        self.acc_MAE        = 0                 # El MAE acumulado
        self.acc_MAE_use    = 0                 # El uso de este neurovector cuando se usa como regresión
        self.acc_MSE        = 0                 # El MSE acumulado
        self.acc_hits       = 0                 # Impactos totales que ha recibido
        self.acc_certainty  = 0                 # La suma de certeza que acumula este neurovector

        # KPIS Calculados
        self.energy     = 0                     # La energia del neurovector según el cálculo
        self.RMSE        = 0                     # El MSE
        self.MAE        = 0                     # El MAE
        self.certainty  = 0                     # La certeza média del neurovector

        # Variables de gobierno
        self.nodes:set[Node]            = set() # El listado de nodos que está asociado a este neurovector
        self.vectors:dict[int,Vector]   = {}    # Los datos de los vectores asociados a los nodos
        self.axons:dict[str,Axon]       = {}    # Los axones que cuelgan del nodo
        self.row        = row                   # La fila del dataset asociada
        self.id         = id                    # El identificador del neurovector
        self.epoch      = epoch                 # La época de entrenamiento
        self.n_axons    = 0                     # Número de axones que tiene el neurovector

    def cal(self,mae:float,mse:float,success:bool,certainty:float,residues:set[Node],hits:set[Node],var:'Var_Study'):
        """
        Actualización y calculo de los KPIS generales del neurovector
        """
        self.acc_use+=1
        self.acc_successes+=success
        self.acc_certainty+=certainty
        self.energy=self.acc_successes*self.acc_successes/self.acc_use
        self.certainty=self.acc_certainty/self.acc_use
        if mae!=None:
            self.acc_MAE_use+=1
            self.acc_MAE+=mae
            self.acc_MSE+=mse
            self.MAE=self.acc_MAE/self.acc_use
            self.RMSE=math.sqrt(self.acc_MSE/self.acc_use)

        # Recogemos el residuo
        for residue in residues:
            residue.acc_residue+=1
            self.vectors[residue.id].acc_residue+=1
            self.axons[residue.column].acc_residue+=1

        # hacemos un bucle por los nodos implicados en la estimulación
        for node in hits:
            node.cal(mae=mae,mse=mse,success=success)
            self.vectors[node.id].cal(mae=mae,mse=mse,success=success)
            self.axons[node.column].cal(mae=mae,mse=mse,success=success)
            self.acc_hits+=1
        # Recogemos la telemetria de la variable de salida
        # Comprobamos que el neurovector tiene un axon correspondiente a la salida
        if var.column in self.axons:
            # Recorremos los nodos del axon y actualimos las variables si las hubiera
            self.axons[var.column].cal(mae=mae,mse=mse,success=success,output=True)
            for node in self.axons[var.column].nodes:
                node.cal(mae=mae,mse=mse,success=success,is_var=True)
                # Actualizmos el vector para ser salida
                self.vectors[node.id].cal(mae=mae,mse=mse,success=success,output=True)



    def show(self):
        sum_vectores=0
        for node,vector in self.vectors.items():
            sum_vectores+=vector.energy

        print(f"NV ({self.id}) Row: {self.row}, Use: {self.acc_use}, Successes: {self.acc_successes}, Energy: {self.energy}, Hits: {self.acc_hits}, Certain: {self.certainty}, MSE: {self.MSE}, MAE: {self.MAE}, Total Nodes: {len(self.nodes)}, Vectors: {sum_vectores}")
        for node  in self.nodes:
            self.vectors[node.id].show()
            node.show()

class Vector:
    """
    Clase que define el objeto Vector entre el nodo y el neurovector
    """
    def __init__(self,name:str,axon:str) -> None:
        # KPIs del Vector
        self.acc_use        = 0                 # El uso del vector
        self.acc_successes  = 0                 # el número de éxitos del vector
        self.acc_MSE        = 0                 # El MSE acumulado
        self.acc_MAE        = 0                 # El MAE acumulado
        self.acc_residue    = 0                 # Cuantas veces ha sido residuo
        self.acc_MAE_use    = 0                 # El uso de este neurovector cuando se usa como regresión
        # KPIs calculados
        self.energy         = 0                 # La energía según el cálculo
        self.RMSE           = 0                 # EL MSE del vector
        self.MAE            = 0                 # El MAE del Vector
        self.axon           = axon              # El id del axon al que pertenece este vector
        # Variables de gobierno
        self.name           = name              # El nombre del vector formado por el id del Axon+el id del nodo
        self.acc_output     = 0                 # Las veces que ha sido salida

    def cal(self,mae:float,mse:float,success:bool,output:bool=False):
        """
        El cálculo de los KPI calculados
        """
        self.acc_use+=1
        self.acc_output+=output
        self.acc_successes+=success
        self.energy=self.acc_successes/self.acc_use
        if mae!=None:
            self.acc_MAE_use+=1
            self.acc_MAE+=mae
            self.acc_MSE+=mse
            self.MAE=self.acc_MAE/self.acc_use
            self.RMSE=math.sqrt(self.acc_MSE/self.acc_use)

    def show(self):
        print (f"   ->Vector ({self.name}): Use: {self.acc_use}, Successes: {self.acc_successes}, Energy: {self.energy}, Residue: {self.acc_residue}, MAE: {self.MAE}, MSE: {self.MSE}")

class Axon:
    """
    Son los canales del neurovector hacia los vectores y Nodos, Corresponden a las columnas o posiciones de un dataset
    """
    def __init__(self,col:str,nv:int) -> None:
        self.column:str             = col    # El nombre de la columna asociado al Axon
        self.nodes:set[Node]        = set() # Los nodos asociados al Axon
        self.vector:set[Vector]     = set() # Los vectores asociados al Axon
        self.acc_use:int            = 0     # El uso de este Axon
        self.acc_successes:int      = 0     # El éxito de este Axon
        self.acc_MAE                = 0     # El acumulado de MAE
        self.acc_MSE               = 0     # el acumulado de RMSE
        self.acc_MAE_use            = 0     # el uso del MAE para poder 
        self.energy:float           = 0     # La energía del Axon
        self.MAE:float              = 0     # El MAE del Axon
        self.RMSE:float             = 0     # El RMSE del Axon
        self.acc_residue            = 0     # Las veces que ha sido residuo
        self.nv:int                 = nv    # El nombre del axon que es igual al nv+' '+columna
        self.acc_output             = 0     # Las veces que ha sido salida
    
    def cal(self,mae:float,mse:float,success:bool,output:bool=False):
        """
        El cálculo de los KPI calculados
        """
        self.acc_use+=1
        self.acc_successes+=success
        self.energy=self.acc_successes/self.acc_use
        self.acc_output+=output
        if mae!=None:
            self.acc_MAE_use+=1
            self.acc_MAE+=mae
            self.acc_MSE+=mse
            self.MAE=self.acc_MAE/self.acc_use
            self.RMSE=math.sqrt(self.acc_MSE/self.acc_use)

class Candidate:
    """
    Clase con toda la telemetria de la propagación del neurovector, almacena el resultado de un alcance
    """
    def __init__(self,nv:Neurovector,
                 knowledge:float=0,
                 tot_hits:int=0,
                 max_hits:int=0,
                 tot_esti:int=0,
                 set_hits:set[Node]={},
                 residues:set[Node]={},
                 unknown:set[Node]={},
                 var:'Var_Study'=None,
                 node:Node=None,scope:int=0
                 ) -> None:
        
        # KPIS de la predicción
        self.tot_hits:int          =   tot_hits     # Impactos en el neurovector del scope
        self.max_hits:int          =   max_hits     # Lo impactos máximos en el neurovector primero
        self.knowledge:float       =   knowledge    # El conocimiento que se tiene del estimulo
        self.tot_esti:int          =   tot_esti     # El número total de estímulos
        self.success:bool          =   None         # Boleano si se ha tenido exito
        self.certainty_rel:float   =   0            # La certeza del candidato relativa
        self.certainty_abs:float   =   0            # La certeza máxima alcanzada en el scope
        
        #KPIS Cálculados
        self.sum_vectors:float     =   0       # La suma de energía de los vectores
        self.sum_nodes:float       =   0       # La suma de energía de los nodos
        self.energy_ponderate:float=   0       # La energía ponderada del neruovector (por definir)
        self.MSE_Nodes:float       =   0       # EL MSE de los nodos implicados
        self.MSE_Vectors:float     =   0       # El MSE de los vectores implicados
        self.MAE_Nodes:float       =   0       # EL MAE de los nodos implicados
        self.MAE_Vectors:float     =   0       # El MAE de los vectores implicados
        self.MAE:float             =   0       # El error absoluto de la predicción
        self.MSE:float             =   0       # El MSE de la predicción
        
        # Variables de gobierno
        self.scope                      = scope     # El número de alcance
        self.var:Var_Study              = var       # La variable a predecir
        self.predict:Node               = node      # El nodo de salida predicho Si procede
        self.neurovector:Neurovector    = nv        # El neurovector usando en el cálculo
        self.node_hits:set[Node]        = set_hits  # Los nodos que se han impactado
        self.residues:set[Node]         = residues  # Los residuos que se han impactado
        self.unknown:set[Node]          = unknown   # Los nodos desconocidos
        self.certainty()
        self.cal()
        self.sum_relation_data()
    
    def certainty(self):
        # Calculamos la certerza
        if self.tot_esti>0:
            self.certainty_rel = self.tot_hits / self.tot_esti
            self.certainty_abs = self.max_hits / self.tot_esti

    def cal(self):
        """
        Realiza diferentes cálculos con el candidato, evalua el éxito y el MAE
        """
        # Comprobamos el exito final de la predicción
        # Si la variable está vacia ponemos a None el éxito
        if not self.var.empty:
            # Sino tenemo predicción nos ponemos a false
            if self.predict==None:
                self.success=False
            else:
                if self.var.value==self.predict.value:
                    self.success=True
                elif self.certainty_rel!=1:
                    # Si el acierto no es 
                    self.success=False
                else:
                    self.success=False
                    #print (f"Discrepancia NV: {self.neurovector.id} con fila {self.neurovector.row}")
                    #self.show()
                    #quit()
                    #raise ValueError("La certeza es del 100% y los datos no son iguales")
                
            # Comprobamos que la variable de estudio es numérica para poder hacer los cálculos de error
            if self.var.numeric:
                # La siguiente comprobación es que si el nodo que tenemos de predicción es numérico y la variable también lo es    
                mae=self.var.value
                if self.predict!=None:    
                    if self.predict.numeric:
                        # Cálculo de MAE y MSE y éxito
                        mae = abs(mae - self.predict.value)
                
                self.MAE=mae
                self.MSE=mae*mae

    def sum_relation_data(self):
        # Inicializimos a 0 los cálculos de los errores de los vectores y nodos
        self.sum_nodes      = 0
        self.sum_vectors    = 0
        self.MAE_Nodes      = 0
        self.MAE_Vectors    = 0
        self.MSE_Nodes      = 0
        self.MSE_Vectors    = 0
        n=0
        if self.neurovector!=None:
            # Recorremos todos los nodos que tenemos de entrada para propagar los calculos hacia la predicción
            for node in self.node_hits:
                n+=1
                self.sum_vectors    += self.neurovector.vectors[node.id].energy
                self.sum_nodes      += node.energy
                self.MAE_Nodes      += node.acc_MAE
                self.MAE_Vectors    += self.neurovector.vectors[node.id].acc_MAE
                self.MSE_Nodes      += node.acc_MSE
                self.MSE_Vectors    += self.neurovector.vectors[node.id].acc_MSE
            if n>0:
                self.MAE_Nodes      = self.MAE_Nodes/n
                self.MSE_Nodes      = self.MSE_Nodes/n
                self.MAE_Vectors    = self.MAE_Vectors/n
                self.MSE_Vectors    = self.MSE_Vectors/n
                self.sum_nodes      = self.sum_nodes/n
                self.sum_vectors    = self.sum_vectors/n

            # Energía ponderada
            self.energy_ponderate=self.neurovector.energy+self.sum_nodes+self.sum_vectors

    def show(self):
        print (" ")
        print (f"PREDICCIÓN: {self.var.column} {self.var.value}, Númerica: {self.var.numeric}, Vacía: {self.var.empty}")
        print ("----------------------------")
        print (f"Certeza: {self.certainty_rel}, Impactos: {self.tot_hits}, Energy Nodes: {self.sum_nodes}, Energy Vectors: {self.sum_vectors} ")
        print (f"Predicción:{self.predict.value}-{self.success}, MAE: {self.MAE}, MSE: {self.MSE}, MAE Nodes: {self.MAE_Nodes}, MAE Vecotors: {self.MAE_Vectors}")

class Var_Study:
    """
    Clase que almacena las variables encontradas en un estimúlo
    """
    def __init__(self,value:str='',column:str='',node:Node=None, empty:bool=True) -> None:
        self.value:str      = value         # El valor de la variable
        self.numeric,_      = UtilityHelper.isNumber(value)            # Si es numérico
        self.node:Node      = node          # El nodo asociado en la variable
        self.column:str     = column        # La columna de la variable
        self.empty:bool     = empty         # Si la variable está vacia

class Cal_Method(ABC):
    def __init__(self) -> None:
        self.name:str       = ''
        self.max:Candidate  = None

    @abstractmethod
    def evaluate_max(self,candidate:Candidate):
        pass

    def reset(self,candidate:Candidate,scope:int=1):
        """
        Realice un reset del cálculo
        """
        self.max=candidate

    def get_mae(self)->tuple[bool,float]:
        """
        Optiene el MAE del candidato seleccionado
        """
        if self.max==None:
            return 0,0,0
        else:
            numeric = self.max.var.numeric
        value=None
        if self.max.predict!=None:
            value=self.max.predict.value
        mae     = 0
        if numeric:
            mae=self.max.MAE

        return numeric, mae,value
    
    def get_success(self)->bool:
        """
        Resuelve si se ha tenido éxito o no, por defecto se coge el resultado del candidato elegido
        """
        if self.max==None:
            success=None
        else:
            success=self.max.success
        return success

class Tokenizer:
    """
    Define el objeto que maneja la tokenización y prograpagación en los neurovectores
    """
    def __init__(self,dic_nodes:dict[str,Node] ,stimulus: pd.Series,vars:list, scope:int) -> None:
        self.find_nodes:set[Node]             = set()     # Los nodos estimulados
        self.rest_stimulus:dict[str,str]      = {}        # El resto de estimulo que no compredemos
        self.find_vars:dict[str,Var_Study]    = {}        # Diccionario con las variables y sus valores que hemos encontrado, además si procede ponemos el Nodo
        self.count_neurovectors               = Counter() # El cálculo de los impactos
        self.total_stimulous:int              = 0         # El número total de estimulos introducidos
        self.str_stimolous:str                = ''        # El estimulo en formato cadena
        self.total_vars:int                   = 0         # El total de las variables
        self.total_nodes:int                  = 0         # El total de los nodos encontrados
        self.knowledge:float                  = 0         # El porcentaje de conocimiento que tenemos sobre el estímulo
        str_vars=''
        for col, value in stimulus.items():
            empty_value = UtilityHelper.isEmpty(value)
            node=None
            if not empty_value:
                node_key = f"{col} {value}"
                node = dic_nodes.get(node_key, None)

            if col in vars:
                self.find_vars[col]=Var_Study(value=value,node=node,column=col,empty=empty_value)
                str_vars+='|+'+col

            elif not empty_value:
                self.total_stimulous+=1
                self.str_stimolous+='|'+col+' '+str(value)
                if node is not None:
                    self.find_nodes.add(node)
                    self.count_neurovectors.update(node.nvs)
                else:
                    # Añadimos lo que no sabemos al resto de estimulos
                    self.rest_stimulus[col]=value
        
        self.str_stimolous=self.str_stimolous[1:]+str_vars
        self.count_neurovectors = self.count_neurovectors.most_common(scope)
        self.total_vars =len(self.find_vars)
        self.total_nodes= len(self.find_nodes)
        if self.total_stimulous>0:
            self.knowledge=self.total_nodes/self.total_stimulous
   

    def show(self):
        print ("TOKENIZACIÓN Y PROPAGACIÓN")
        print ("--------------------------")
        print (f"Estimulo de entrada: {self.str_stimolous}")
        print (f"Estimulos que no se comprenden: {self.rest_stimulus}")
        print (f"Total estímulo: {self.total_stimulous}, Variables: {self.total_vars}, Nodos encontrados: {self.total_nodes}")
        print (f"Neurovectores: {self.count_neurovectors}")
        print (f"Grado de conocimiento: {(self.knowledge*100):.2f}")

class Result:
    """
    Clase que contiene las columnas de los resultados
    """
    def __init__(self,index_row:int=0,candidate:Candidate=None,method:str='') -> None:
        self.row:int        = index_row
        self.expected       = 0
        self.predict        = None
        self.success        = None
        self.explication    = None
        self.nv             = None
        self.certainty      = 0
        self.cert_dif_max   = 0
        self.MAE            = 0
        self.hits           = 0
        self.energy         = 0
        self.vectors        = 0
        self.nodes          = 0
        self._candidate                 = candidate
        self._detail:list[Candidate]    = []
        self._method        = method

    def to_dict(self, exception: str = None):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') or k == exception
        }
    
    def set_detail(self,detail:list):
        self._detail=detail
