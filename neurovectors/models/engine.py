# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
#
# This software is provided solely for non-commercial use. 
# Refer to the LICENSE file for full terms and conditions.         
 
from .analytics import Analytics
from .graph import Neurovector,Node,Vector,Candidate,Tokenizer,Axon,Result
from ..utils import PaintNV
from pandasgui import show
import pandas as pd
import pickle

class Engine:
    def __init__(self, verbose: bool = True, name:str=''):
        """
        Parent class that defines the methods that every engine must have.
        :param verbose: If True, the engine will print status messages.
        """
        self.verbose:bool       = verbose
        self.name               = name
        # Nodes
        self.nodes: dict[int, Node]         = {}    # Diccionario de Nodos
        self.index_node: dict[int, Node]    = {}    # Índice id_nodo > id: nodes
        self.pointer_node: int = 0                  # Node ID pointer
        
        # Neurovectrors
        self.nvs: dict[int,Neurovector]     = {}    # El diccionario de neurovectores
        self.pointer_nv: int                = 0     # Neurovector ID pointer
        self.count_vectors:int              = 0     # Infomración sobre el número de vectores
        self.count_axons:int                = 0     # Información sobre el número de axones

    def _create_node(self, column: str, value: any) -> Node:
        """
        Creates a node in the nodes dictionary. If the node already exists, returns its ID.
        :param column: The column name.
        :param value: The value associated with the column.
        :return: The object Node of the created or existing node, or False if invalid.
        """
        if value!=value or value == '':           # Comprobamos que tengemos algún dato en el valor
            return False

        node_key = f"{column} {value}"
        node = self.nodes.get(node_key, None)
        if node is None:
            node_id = self.pointer_node
            node = Node(id=node_id,column=column,value=value,name=node_key)
            self.nodes[node_key] = node
            self.index_node[node_id] = node
            self.pointer_node += 1

        return node

 

    def _find_inset(self, set_study: set[Node], var, property_name: str) -> Node:
        """
        Encuentra un Nodo en `set_study` según el valor de una propiedad especificada.

        :param set_study: Conjunto de nodos en el que buscar.
        :param var: Valor de la propiedad a buscar.
        :param property_name: Nombre de la propiedad en el objeto `Node` por la que buscar.
        :return: El nodo encontrado o None si no existe.
        """
        objeto_encontrado = next((obj for obj in set_study if getattr(obj, property_name, None) == var), None)
        return objeto_encontrado

    def conver_row(self, index: int, row: pd.Series,epoch:int=0) -> bool:
        """
        Converts a row into a Neurovector and updates internal structures.
        :param index: The index of the row in the dataset.
        :param row: A pandas Series representing the row to convert.
        :return: True si se ha convertido correctamente la fila o False si no hay datos
        """
        # Create the Neurovector ID
        id_nv=self.pointer_nv
        nv=Neurovector(row=index,id=id_nv,epoch=epoch)
        real=False
        for col, value in row.items():
            # 1 - Tokenize the value
            node = self._create_node(col, value)
            if node:
                # 2 - Create the relationship (nv->id_nodo,nv->vector, nodes->nv, Axones)
                id_node=node.id
                nv.vectors[id_node]=Vector(name=f"{id_nv}-{id_node}",axon=col)
                axon=Axon(col=col,nv=id_nv)
                axon.nodes.add(node)
                axon.vector.add(nv.vectors[id_node])
                nv.axons[col]=axon
                nv.n_axons+=1
                self.count_axons+=1
                self.count_vectors+=1
                nv.nodes.add(node)
                node.nvs.add(id_nv)
                # Es un neurovector real
                real=True
        if real:        
            # FInalmente creamos el neurovector con los vectores añadidos
            self.nvs[id_nv]=nv
            # Increment the Neurovector ID pointer
            self.pointer_nv += 1
        
        return real
    
    def add_nodes_to_nv(self,result:Result):
        """
        Añade un set de nodos al neurovector seleccionado
        """
        id_nv=result.nv
        nv=result._candidate.neurovector
        if id_nv!=None and len(result._candidate.unknown)>0  and len(result._candidate.unknown)<2:
            for node in result._candidate.unknown:
                #Create the relationship (nv->id_nodo,nv->vector, nodes->nv, nv->Axon)
                id_node=node.id
                if node.column==result._candidate.var.column:
                    print ("Esto es una variable...")

                if node.column not in nv.axons:
                    self.count_axons+=1
                    nv.axons[node.column]=Axon(col=node.column,nv=id_nv)
                    nv.n_axons+=1
                if id_node in nv.vectors:
                    print ("Error el id de nodo ya está en los vectores de este neurovector")
                    
                if id_node in nv.nodes:
                    print ("Confirmamos")

                nv.vectors[id_node]=Vector(f"{id_nv}-{id_node}",axon=node.column)
                self.count_vectors+=1
                nv.nodes.add(node)
                node.nvs.add(id_nv)
                nv.axons[node.column].nodes.add(node)
                nv.axons[node.column].vector.add(nv.vectors[id_node])
           
    def predict(self,analitycs:Analytics, stimulus:str = '', scope:int = 1,mode:bool=True,index_row:int=0) -> Result:
        """
        Performs a prediction based on the stimulus. The stimulus can be a pandas Series or a string.
        :param stimulus: The input stimulus, can be a string or a pandas Series.
        :param scope: The scope of the prediction (number of top neurovectors to consider).
        :return: - results: List of dictionary of predicction.       
        """
        # Initialize variables       
        deep:int = 0
        # Sets de manejo de propagación
        set_arrival:  set[Node] = set()                   # Los Nodos de llegada alcanzador por el neurovector a estudiar
        set_hits:     set[Node] = set()                   # Los Nodos que hemos encontrado del estimulo
        set_study:    set[Node] = set()                   # Los Nodos que son objeto del studio para alcanzar la predicción (Variables + Residuo)
        # 1 - Tokenizar y propagación
        propagation = Tokenizer(dic_nodes=self.nodes,stimulus=stimulus,vars=analitycs.vars,scope=scope)
        # Comprobamos que hemos encontrado estimúlos, sino nos salimos
        if propagation.total_nodes>0:
            # Iniciamos el análisis de los neurovectores seleccionados
            max_hits=propagation.count_neurovectors[0][1]
            for neurovector in propagation.count_neurovectors:
                # Cogemos el objeto neurovector y los impactos que ha tenido
                object_nv   = self.nvs[neurovector[0]]
                hits        = neurovector[1]
                # 2 - Propagation - Nodes reached in the neurovector also include the stimulated ones
                set_arrival = object_nv.nodes
                # La lista de nodos que ha entendido / encontrado (Nodos relacionados con el NV - Nodos Estimulados)
                set_hits = set_arrival.intersection(propagation.find_nodes)
                # La lista de nodos que son objeto de estudio
                set_study = set_arrival.difference(propagation.find_nodes)
                # Nodos que hemos encontrado pero que no están en el neurovector
                set_unknown = propagation.find_nodes.difference(set_arrival)             
                # Hacemos el bucle por las variables de estudio para comparar los resultados
                for var,data in propagation.find_vars.items():
                    # Buscamos la predicción en los resultados por el valor de la columna en los nodos de llegada
                    # prediction  = self._find_inset(set_study=set_study,var=var,property_name='column')
                    if var not in object_nv.axons:
                        prediction=None
                    else:
                        prediction = next(iter(object_nv.axons[var].nodes))
                        # Quitamos la salida del estudio para dejar solo el residuo
                        set_study.discard(prediction)

                    new_candidate=Candidate(nv=object_nv,knowledge=propagation.knowledge,
                                            tot_hits=hits,max_hits=max_hits,tot_esti=propagation.total_stimulous,
                                    set_hits=set_hits, residues=set_study,unknown=set_unknown,var=data,node=prediction,scope=deep)
                    
                    analitycs.add_candidate(new_candidate)
            
                # Control del alcance
                deep+=1
        else:
            for var,data in propagation.find_vars.items():
                new_candidate=Candidate(nv=None,tot_hits=0,max_hits=0,tot_esti=propagation.total_stimulous,
                                        set_hits=set(),residues=set(),var=data,node=None,scope=0)
                analitycs.add_candidate(candidate=new_candidate)
        
        return analitycs.cal_prediction(mode,index_row=index_row)

    def save(self, file: str) -> None:
        """
        Saves the Neurovectors to a file using pickle.

        :param file: The file path where Neurovectors will be saved.
        :return: None
        """
        with open(file, 'wb') as fichero:
            pickle.dump({
                'nodes': self.nodes,
                'index_node': self.index_node,
                'pointer_node': self.pointer_node,
                'node_nv_index': self.node_nv_index,
                'neurovectors': self.nvs,
                'pointer_nv': self.pointer_nv,
                'count_vectors': self.count_vectors,
                'count_axons': self.count_axons
            }, fichero)
        
        if self.verbose:
            print(f"Variables saved to {file}")

    def load(self, file: str) -> None:
        """
        Loads the Neurovectors from a file using pickle.

        :param file: The file path from where Neurovectors will be loaded.
        :return: None
        """
        with open(file, 'rb') as fichero:
            datos = pickle.load(fichero)
            self.nodes = datos['nodes']
            self.index_node = datos['index_node']
            self.pointer_node = datos['pointer_node']
            self.node_nv_index = datos['node_nv_index']
            self.nvs = datos['neurovectors']
            self.pointer_nv = datos['pointer_nv']
            self.count_vectors=datos['count_vectors']
            self.count_axons=datos['count_axons']
        if self.verbose:
            print(f"Variables loaded from {file}")

    def convert_stimulus(self,stimulus: any = '',sep:str='|')->pd.Series:
        """
        Convierte un stimulo a una serie de panda
        """
        if isinstance(stimulus, pd.Series):
            return stimulus
        
        if isinstance(stimulus, str):
            elements = stimulus.split(sep)
            data_dict = {}
            for element in elements:
                parts = element.split(" ", 1)
                if len(parts) == 2:
                    column, value = parts
                    data_dict[column] = value if value else None
                else:
                    # Si no hay valor después de la columna, guardamos `None`
                    data_dict[parts[0]] = None

            # Convertimos el diccionario en una Series
            stimulus = pd.Series(data_dict)
        else:
            raise ValueError("Error in stimulus format")
        
        return stimulus

    def find_node(self, col: str, value: str) -> Node:
        """
        Searches for a node by column and value.

        :param col: The column name.
        :param value: The value to search for.
        :return: A Object node if found, otherwise None.
        """
        if value!=value or value == '':
            return None
        node_key = f"{col} {value}"
        return self.nodes.get(node_key, None)
    
    def find_id_node(self, id: int) -> int:
        """
        Returns the node name given a node ID.

        :param id: The node ID.
        :return: The node name if found, otherwise None.
        """
        return self.index_node.get(id, None)

    def show_info(self):
         """
         Muestra la información principal del motor
         """
         print ("-----------")
         print (f"Motor: {self.name}, Neurovectors: {self.pointer_nv}, Nodes: {self.pointer_node}, Vectors: {self.count_vectors}, Axons: {self.count_axons}")

    def show_all(self):
        """
        Muestra la información de los nodos
        """
        dic_nodes_inputs:dict={}
        dic_nodes_outputs:dict={}
        dic_nv:dict={}
        dic_df={}
        dic_vector_inputs={}
        dic_vector_outpus={}
        dic_axons_inputs={}
        dic_axons_outputs={}
        # Diccionario con los neurovectores
        for id,nv in self.nvs.items():
            dic_nv[nv.id]={'row':nv.row,
                           'use':nv.acc_use,
                           'hits':nv.acc_hits,
                           'success':nv.acc_successes,
                           'energy':nv.energy,
                           'certainty':nv.certainty,
                           'MAE':nv.MAE,
                           'RMSE':nv.RMSE,
                           'Axons':nv.n_axons,
                           'Vectors':len(nv.vectors)
            }
            for id_vector,vector in nv.vectors.items():
                if vector.acc_output>0:
                    dic_vector_outpus[vector.name]={'use':vector.acc_use,
                                            'success':vector.acc_successes,
                                            'energy':vector.energy,
                                            'MAE':vector.MAE,
                                            'RMSE':vector.RMSE,
                                            'residue':vector.acc_residue
                                            }
                else:
                    dic_vector_inputs[vector.name]={'use':vector.acc_use,
                                            'success':vector.acc_successes,
                                            'energy':vector.energy,
                                            'MAE':vector.MAE,
                                            'RMSE':vector.RMSE,
                                            'residue':vector.acc_residue
                                            }
                    
            for id_axon,axon in nv.axons.items():
                if axon.acc_output>0:
                    dic_axons_outputs[str(axon.nv)+' '+axon.column]={'column':axon.column,
                                                         'use':axon.acc_use,
                                                         'success':axon.acc_successes,
                                                         'residue':axon.acc_residue,
                                                         'MAE':axon.MAE,
                                                         'RMSE':axon.RMSE,
                                                         'energy':axon.energy,
                                                         'nodes':len(axon.nodes),
                                                         'vectors':len(axon.vector)
                    }
                else:
                    dic_axons_inputs[str(axon.nv)+' '+axon.column]={'column':axon.column,
                                                         'use':axon.acc_use,
                                                         'success':axon.acc_successes,
                                                         'residue':axon.acc_residue,
                                                         'MAE':axon.MAE,
                                                         'RMSE':axon.RMSE,
                                                         'energy':axon.energy,
                                                         'nodes':len(axon.nodes),
                                                         'vectors':len(axon.vector)
                    }    


        # Diccionarios con los nodos
        for name,node in self.nodes.items():
            if node.acc_use_var>0:
                dic_nodes_outputs[node.id]={'name':node.name,
                                           'column':node.column,
                                           'value':node.value,
                                           'numeric':node.numeric,
                                           'use':node.acc_use_var,
                                           'use_MAE':node.acc_MAE_use,
                                           'success':node.acc_successes,
                                           'energy':node.energy,
                                           'residues':node.acc_residue,
                                           'MAE':node.MAE,
                                           'RMSE':node.RMSE
                                           }
            else:    
                dic_nodes_inputs[node.id]={'name':node.name,
                                           'column':node.column,
                                           'value':node.value,
                                           'numeric':node.numeric,
                                           'use':node.acc_use,
                                           'use_MAE':node.acc_MAE_use,
                                           'success':node.acc_successes,
                                           'energy':node.energy,
                                           'residues':node.acc_residue,
                                           'MAE':node.MAE,
                                           'RMSE':node.RMSE
                                           }
                
        # Convertir los diccionarios a DataFrames
        dic_df['Inputs'] = pd.DataFrame.from_dict(dic_nodes_inputs, orient='index')
        dic_df['Outputs']  = pd.DataFrame.from_dict(dic_nodes_outputs, orient='index')
        dic_df['Neurovector'] = pd.DataFrame.from_dict(dic_nv, orient='index')
        dic_df['Vectors_Inputs'] = pd.DataFrame.from_dict(dic_vector_inputs, orient='index')
        dic_df['Vectors_Outputs'] = pd.DataFrame.from_dict(dic_vector_outpus, orient='index')
        dic_df['Axons_Inputs'] = pd.DataFrame.from_dict(dic_axons_inputs, orient='index')
        dic_df['Axons_Outputs'] = pd.DataFrame.from_dict(dic_axons_outputs, orient='index')
        show(**dic_df)
  
    def paint_nv(self,nv:int=0):
        if self.verbose:
            print (f"Pintando neurovector {nv}")
        nv_object=self.nvs[nv]
        paint=PaintNV()
        paint.draw_neurovector(neurovector=nv_object)
        paint.show()

# END
