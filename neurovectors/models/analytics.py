# © 2025. Software anonimizado. Todos los derechos reservados.
# Consulte LICENSE para términos y condiciones.

from .graph import Candidate,Var_Study, Cal_Method, Result
import importlib
import numpy as np
import math
import json
from tabulate import tabulate
import pandas as pd
from pandasgui import show
import warnings
import matplotlib.pyplot as plt
# Ignorar específicamente el FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)



class Output:
    def __init__(self,list_methods:list,name_var:str='',sel_method:str='',scope:int=1,map:dict[str,str]=None)->None:
        """
        El espacio de trabajo por variable
        """
        self.list_methods:list                          = list_methods  # La lista de métodos que usaremos
        self.methods:dict[str,Method]                   = {}            # El diccionario de métodos
        self.last_var:Var_Study                         = None          # Los datos de la última variable de estudio que hemos usado
        self.name_var:str                               = name_var      # El nombre de la variable que está asociada a este espacio
        self.sel_method:str                             = sel_method    # El método que hemos seleccionado
        self.scope:int                                  = scope         # El alcance para calcular en el algoritmo
        self.methods_map:dict[str,str]                  = map           # El mapa de certezas y métodos
        self.use                                        = 0             # El uso final para el calculo de precisión
        self.use_MAE                                    = 0             # El uso de MAE
        self.success                                    = 0             # Par el calculo de la precisión
        self.accuracy                                   = 0             # La precisión final
        self.RMSE                                       = 0             # El RMSE Final
        self.MAE                                        = 0             # EL MAE Final
        self.final_results:list[Result]                 = []            # La lisa de resultados globales de cada predicción
        # Creamos el espacio de trabajo de los métodos
        self.methods = {name_method: Method(name=name_method,scope=scope) for name_method in self.list_methods}
                
    def new_candidate(self,reset:bool,candidate:Candidate)->None:
        """
        Añadimos un candidato y reiniciamos en el caso que así se indique
        """
        # Iteramos por todos los métodos para que hagan sus apuestas sobre el candidato
        for method in self.methods.values():
            method.add_candidate(reset=reset,candidate=candidate)

    def cal_prediction(self,mode:bool,index_row:int)->Result:
        """
        Cerramos la predicción con el campo calculado
        """        
        self.use+=1
        final_result=None 
        total_result:dict[str,Result]={}
        for name,method in self.methods.items():
            is_select=(name==self.sel_method)  
            result=method.recolet_kpis(mode=mode*is_select,index_row=index_row)
            total_result[name]=result
            
        if self.methods_map is None:  
            seleccion=self.sel_method
        else:
            certainty=total_result[self.sel_method].certainty
            bind=int(round(certainty*10,0))*10
            if bind in self.methods_map:
                seleccion=self.methods_map[bind]
            else:
                seleccion=self.sel_method
  
        final_result=total_result[seleccion]
        final_result._method=seleccion
        self.success+=total_result[seleccion].success
        self.MAE+=total_result[seleccion].MAE
        self.RMSE+=(total_result[seleccion].MAE*total_result[seleccion].MAE)
        self.final_results.append(final_result)
        return final_result
    
    def get_results(self,sel_method:str=''):
        if sel_method not in self.list_methods and sel_method!='':
            raise ValueError("El método no esta en la lista de métodos declarados")
        if sel_method=='':
            sel_method=self.list_methods
        else:
            sel_method=[sel_method]        
        
        df = pd.DataFrame([obj.to_dict(exception='_method') for  obj in self.final_results])
        dic_df={}
        dic_df['Final_results']=df
        for method in sel_method:
            dic_df[method]=self.methods[method].get_result()
    
        show(**dic_df)

    def show_curves(self,sel_method:str=''):
        if sel_method not in self.list_methods and sel_method!='':
            raise ValueError("El método no esta en la lista de métodos declarados")
        if sel_method=='':
            sel_method=self.list_methods
        else:
            sel_method=[sel_method]        

        for method in sel_method:
            self.methods[method].show_curve()

    def runner_curves(self,verbose:bool=True):
        """
        Crea un listado de certezas y cual es el algoritmo mejor que se adapta en ese porcentaje
        """
        dic_df_curve={}
        primero=self.sel_method
        dic_df_curve[f"{primero}"]=''
        for method in self.list_methods:
            dic_df_curve[f"{method}"]=self.methods[method].get_curves()

        
        # Crear un DataFrame para almacenar los resultados
        certezas = dic_df_curve[primero]["Certeza (%)"]  # Todas las certezas deben coincidir entre DataFrames
        resultados = pd.DataFrame({"Certeza (%)": certezas})

        # Encontrar el DataFrame con la máxima precisión para cada certeza
        mejor_dataframe = []
        ganancia_precision = []
        ganancia_ponderada = []

        for certeza in certezas:
            # Obtener las precisiones de cada DataFrame para la certeza actual
            precisiones = {key: df[df["Certeza (%)"] == certeza]["Precisión (%)"].values[0] for key, df in dic_df_curve.items()}
            volumenes = {key: df[df["Certeza (%)"] == certeza]["Volumen (%)"].values[0] for key, df in dic_df_curve.items()}
            # Determinar el DataFrame con la máxima precisión
            mejor = max(precisiones, key=precisiones.get)
            mejor_dataframe.append(mejor)

            # Calcular la ganancia de precisión respecto a "A"
            ganancia = precisiones[mejor] - precisiones[primero]
            ganancia_precision.append(ganancia)
            
            # Calcular la ganancia ponderada si hay volumen total
            total_volumen = sum(volumenes.values())
            if total_volumen > 0:
                volumen_a = volumenes[primero]
                ganancia_pond = ganancia * (volumen_a / total_volumen)
            else:
                ganancia_pond = 0  # Si no hay volumen, la ganancia ponderada es 0
            ganancia_ponderada.append(ganancia_pond)

        # Agregar la columna de resultados al DataFrame final
        resultados["Mejor Metodo"] = mejor_dataframe
        resultados["Ganancia Precisión (%)"] = ganancia_precision
        resultados["Ganancia Ponderada"] = ganancia_ponderada
        # Calcular la media ponderada final
        media_ponderada_final = sum(ganancia_ponderada)
        # Mostrar el DataFrame final
        if verbose:
            print(resultados)
            print(f"\nMedia ponderada de ganancia de precisión respecto a {primero}: {media_ponderada_final:.2f}%")
        
        return resultados,media_ponderada_final

class Method:
    def __init__(self,name:str,scope:int=1)->None:
        """
        Todas la telemetría necesaria para hacer los cálculos globales de precisión y error
        """
        self.name:str            = name
        self.acc_MSE             = 0     # Acumulador de MSE usando clasificación, para calcularlo de manera global
        self.acc_MAE             = 0     # Acumulador de MAE para calcularlo de manera global
        self.acc_use             = 0     # El acumulador de uso del analytics de manera global
        self.acc_success         = 0     # El acumulador de éxito
        self.acc_use_MAE         = 0     # El acumulador de uso del analytics cuando calcula MAE de manera global
        self.curve_kwnow:np[float] = np.zeros((11,2),dtype=int)   # La curva de análisis del conocimiento
        # Variables finales calculadas
        self.accuracy            = 0     # La precisión calculada acc_success/acc_use
        self.RMSE                = 0     # El RMSE global 
        self.MAE                 = 0     # EL MAE global

        self.results:list[Result]           = []      # La lisa de resultados globales de cada predicción

        # Parámetros de calculo en la predicción
        self.cal:Cal_Method             = self._load_method(name=name)  # El objeto de cálculo
        self.scope:int                  = scope

    def _load_method(self,name:str)->Cal_Method:
        # Primero intentamos cargar desde el nuevo paquete
        primary_module = f"neurovectors.methods.{name}_method"
        fallback_module = f"Alneux.methods.{name}_method"
        last_err = None
        for module_name in (primary_module, fallback_module):
            try:
                module = importlib.import_module(module_name)
                class_name = name
                object = getattr(module, class_name)
                return object()
            except (ModuleNotFoundError, AttributeError, TypeError) as e:
                last_err = e
                continue
        raise ImportError(f"Error al cargar el método '{name}' desde '{primary_module}' o fallback '{fallback_module}': {last_err}")

    def add_candidate(self,reset:bool,candidate:Candidate):
        """
        Evaluamos si el candidato que nos mandan es mejor que el que tenemos ya seleccionado
        """
        if reset:
            self.cal.reset(candidate=candidate,scope=self.scope)
        else:
            self.cal.evaluate_max(candidate=candidate)

    def recolet_kpis(self,mode:bool,index_row:int):
        """
        Calcula varios parámetros de los KPI Locales como el valor médio ponderado
        Cerramos el alcance y establecemos el éxito
        """
        # Aumentamos el uso en los acumuladores
        self.acc_use+=1
        # Ponemos las variables por defecto para usar en caso que no haya datos
        success=None
        bind=0
        # Calculamos el MAE y el MSE
        use,mae,value=self.cal.get_mae()
        mse=0
        if mae!=None:
            mse=mae*mae
            self.acc_use_MAE+=use
            self.acc_MAE+=mae
            self.acc_MSE+=mse
        # Calculamos el éxito
        success=self.cal.get_success()
        # Actualizamos la agrupación de certeza donde almacenaremos los datos para la curva, sino cogemos certeza 0
        if self.cal.max==None:
            certainty=0
        else:
            certainty=self.cal.max.certainty_abs
        bind=int(round(certainty*10,0))
        if success!=None:
            # Actualizamos contadores y KPIS
            self.acc_success+=success
            # Actualizamos la curva de precisión / certeza, en este caso solo cogemos cuando hayamos tenido certeza
            self.curve_kwnow[bind][0]+=success
            self.curve_kwnow[bind][1]+=1
            # Si estamos en modo de entrenamiento entonces actualizamos todos los datos a la red, claro está que tengamos resultados para actualizar
            nv=self.cal.max.neurovector  # Recogemos el neurovector asociado en el resultado
            if mode and nv!=None:
                # Actualizamos los valores del neurovector
                nv.cal(mae=mae,mse=mse,success=success,certainty=certainty,
                       residues=self.cal.max.residues,hits=self.cal.max.node_hits,
                       var=self.cal.max.var)
                
        # else:
        #     raise ValueError("No podemos establecer el éxito en la propagación")

        result=self.add_prediction(mae=mae,success=success,predict_value=value,index_row=index_row)
        return result
        
    def calculate(self):
        """
        Calcula todos los KPIS globales del análisis
        """
        self.accuracy    =   self.acc_success/self.acc_use
        if self.acc_use_MAE==0:
            self.MAE=0
            self.RMSE=0
        else:
            self.MAE         =   self.acc_MAE/self.acc_use_MAE
            self.RMSE        =   math.sqrt(self.acc_MSE/self.acc_use_MAE)

    def add_prediction(self,mae,success,predict_value,index_row:int=0) -> "Result":
        """
        Añadimos un resultado a la lista
        """
        result=Result(index_row=index_row,candidate=self.cal.max)
        if self.cal.max.neurovector!=None:
            result.explication=self.cal.max.neurovector.row
            result.certainty=round(self.cal.max.certainty_abs,2)
            result.cert_dif_max=round(self.cal.max.certainty_abs-self.cal.max.certainty_rel,2)
            result.nv=self.cal.max.neurovector.id
            result.energy=f"{self.cal.max.neurovector.energy:.2f}"
            result.vectors=f"{self.cal.max.sum_vectors:.2f}"
            result.nodes=f"{self.cal.max.sum_nodes:.2f}"
            result.predict=predict_value
        
        result.expected=self.cal.max.var.value
        result.success=success
        result.MAE=mae
        result.hits=self.cal.max.tot_hits

        self.results.append(result)
        return result

    def get_result(self):
        df = pd.DataFrame([obj.to_dict() for  obj in self.results])
        return df
    
    def show_curve(self):
        # Inicializar arreglos para precisiones y volumen de datos
        precisiones = np.zeros(11, dtype=float)
        volumen_porcentaje = np.zeros(11, dtype=float)
        total_datos = np.sum(self.curve_kwnow[:, 1])  # Total de predicciones en todas las categorías

        # Calcular precisiones y volumen de datos donde hay predicciones
        for certeza in range(11):
            aciertos = self.curve_kwnow[certeza, 0]
            total_predicciones = self.curve_kwnow[certeza, 1]
            
            if total_predicciones > 0:  # Solo calcular si hay datos en la categoría
                precisiones[certeza] = (aciertos / total_predicciones) * 100
                volumen_porcentaje[certeza] = (total_predicciones / total_datos) * 100 if total_datos > 0 else 0

        # Filtrar valores de precisión y certeza donde haya datos
        precisiones_con_datos = precisiones[volumen_porcentaje > 0]
        certezas_con_datos = np.arange(0, 101, 10)[volumen_porcentaje > 0]
        volumen_con_datos = volumen_porcentaje[volumen_porcentaje > 0]

        # Calcular precisión media excluyendo los valores sin datos
        precision_media = np.mean(precisiones_con_datos)

        # Calcular certeza media ponderada
        certeza_media = np.average(certezas_con_datos, weights=volumen_con_datos)

        #print("\nCURVA DE CERTEZA-PRECISIÓN:")
        #print(f"{precisiones_con_datos}")

        # Configurar la gráfica
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Graficar la curva de precisión
        ax1.plot(certezas_con_datos, precisiones_con_datos, color='blue', label="Precisión (%)")
        ax1.set_xlabel("Certeza (%)")
        ax1.set_ylabel("Precisión (%)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Graficar la línea de precisión media
        ax1.axhline(y=precision_media, color='green', linestyle='--', label=f"Precisión media ({precision_media:.2f}%)")

        # Graficar la línea de certeza media
        ax1.axvline(x=certeza_media, color='purple', linestyle='--', label=f"Certeza media ({certeza_media:.2f}%)")

        # Configurar un segundo eje y para las barras de volumen de datos
        ax2 = ax1.twinx()
        ax2.bar(certezas_con_datos, volumen_con_datos, color='gray', alpha=0.3, width=8, label="Volumen de datos (%)")
        ax2.set_ylabel("Volumen de datos (%)", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, 100)  # Escala de 0 a 100% para el volumen de datos

        # Añadir leyendas y título
        fig.suptitle(f"Curva de Precisión / Certeza usando '{self.name}'")
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        ax1.grid()

        # Mostrar la gráfica
        plt.show()
        #plt.pause(0.001)

    def get_curves(self):
       # Inicializar arreglos para precisiones y volumen de datos
        precisiones = np.zeros(11, dtype=float)
        volumen_porcentaje = np.zeros(11, dtype=float)
        total_datos = np.sum(self.curve_kwnow[:, 1])  # Total de predicciones en todas las categorías

        # Calcular precisiones y volumen de datos donde hay predicciones
        for certeza in range(11):
            aciertos = self.curve_kwnow[certeza, 0]
            total_predicciones = self.curve_kwnow[certeza, 1]
            
            if total_predicciones > 0:  # Solo calcular si hay datos en la categoría
                precisiones[certeza] = (aciertos / total_predicciones) * 100
                volumen_porcentaje[certeza] = (total_predicciones / total_datos) * 100 if total_datos > 0 else 0

        # Filtrar valores de precisión y certeza donde haya datos
        #precisiones_con_datos = precisiones[volumen_porcentaje > 0]
        #certezas_con_datos = np.arange(0, 101, 10)[volumen_porcentaje > 0]
        #volumen_con_datos = volumen_porcentaje[volumen_porcentaje > 0]

        # Calcular precisión media excluyendo los valores sin datos
        #precision_media = np.mean(precisiones_con_datos)

        # Calcular certeza media ponderada
        # certeza_media = np.average(certezas_con_datos, weights=volumen_con_datos)
        
        # Crear un DataFrame para exportar las curvas
        curve_df = pd.DataFrame({
            "Certeza (%)": np.arange(0, 101, 10),
            "Precisión (%)": precisiones,
            "Volumen (%)": volumen_porcentaje
        })

        return curve_df

class Analytics:
    def __init__(self,verbose:bool=True,method:str='Hits',name:str='',methods_map:dict=None)->None:
        """
        Clase principal del analizador, ejecuta varios métodos de cálculo por variable de salida
        """
        self.verbose:bool                = verbose
        self.list_methods:list           = ['Hits','Avg_predict','Vectors_Energy','Nodes_Energy','Avg','Hybrid_Score','Super_X']
        #self.list_methods:list           = ['Hits']
        self.sel_method                  = method                             # La clave estadísticas por la que vamos a vasar el éxito
        self.name:str                    = name                               # El nombre del Analytics
        self.vars:list                   = ["Not_Vars"]                       # La lista de variables
        self.methods_map:dict[str,pd.DataFrame] = methods_map                 # El diccionario con los rangos de métodos
        self.scope:int                   = 1                                  # El alcanze para el cálculo medio
        if method not in self.list_methods:
            raise ValueError("El método seleccionado no está en la lista de declarados")
        self.new_analytics()

    def set_vars(self,list_vars:list):
        """
        Establece las variables de estudio. En caso contrario pone por defecto una variable dummy
        """
        if len(list_vars)==0:
            list_vars.append("Not_Vars")
            if self.verbose:
                print (f"->> No hay variables definidas, se procede a una propagación total...")
        
        self.vars=list_vars

    def set_methods_map(self,methods_map:dict={},import_json:str=''):
        """
        Establece el mapa de métodos
        """
        self.methods_map:dict[str,pd.DataFrame]={}
        if self.verbose:
            print (f"->Establecido el mapa de métodos")
        if len(methods_map)>0:
            self.methods_map:dict[str,pd.DataFrame]=methods_map
        elif import_json=='':
            raise ValueError(f"No has establecido un fichero json para importar")
        else:
            # Definición de las columnas esperadas
            expected_columns = ["Certeza (%)", "Mejor Metodo"]
            # Código para importar el JSON y recuperar tanto los datos como la variable
            try:
                with open(import_json, "r") as f:
                    imported_data = json.load(f)
                
                # Recuperar los datos del DataFrame
                imported_df = pd.DataFrame(imported_data["map"])
                # Recuperar la variable adicional
                imported_var = imported_data["var"]
                # Verificaciones
                imported_columns = imported_df.columns.tolist()
                columns_match = set(expected_columns) == set(imported_columns)
                columns_message = "Las columnas coinciden." if columns_match else "Las columnas no coinciden."

                imported_rows = imported_df.shape[0]
                # Verificar el número de filas importadas
                expected_rows = 11  # Cambiar este valor si el número esperado es diferente
                rows_match = expected_rows == imported_rows
                rows_message = "El número de filas es correcto." if rows_match else "El número de filas no coincide."
                
                # Mostrar resultados
                if self.verbose:
                    print("Importación exitosa.")
                    print(columns_message)
                    print(rows_message)
                    print(f"Variable adicional importada: {imported_var}")
                    print("El mapa de métodos importado:")
                    print(imported_df)

                self.methods_map[imported_var]=imported_df

            except ValueError as ve:
                print("Error en el formato del archivo JSON:", ve)
            except FileNotFoundError as fnfe:
                print("Archivo no encontrado:", fnfe)
            except Exception as e:
                print("Se produjo un error inesperado:", e)

    def new_analytics(self):
        """
        Inicializamos un nuevo entorno de trabajo del analizador
        """
        if self.verbose:
            print (f"Nuevo espacio de trabajo del analizador: {self.name}")
            if self.methods_map is not None:
                print (f"->Gargado mapa de métodos")
            else:
                print (f"Método: {self.sel_method}")
        self.blocked                            = False                     # Control de flujo para finalizar el acumulado de la predicción
        self.id_predict                         = 0                         # El id del analytic donde llegamos
        self.detail:list[list[Candidate]]       = [[]]                      # La lista de detalle con todos los alcances en la predicción
        self.outputs:dict[str,Output]           = {}                        # El espacio de trabajo por Variable                 
        # Creamos el espacio de trabajo por variable
        for var in self.vars:
            if self.methods_map is not None and var in self.methods_map:
                df=self.methods_map[var]
                certeza_dict = dict(zip(df["Certeza (%)"], df["Mejor Metodo"]))
            else:
                certeza_dict = None
            self.outputs[var] = Output(self.list_methods,name_var=var,sel_method=self.sel_method,scope=self.scope,map=certeza_dict)  

    def add_candidate(self,candidate:Candidate):
        """
        Añade un candidato, analiza el mejor usando los métodos establecidos
        """          
        # Evaluamos el candidato y lo añadimos

        for workspace in self.outputs.values():
            workspace.new_candidate(reset=not self.blocked,candidate=candidate)
        self.blocked=True
        self.detail[self.id_predict].append(candidate)

    def cal_prediction(self,mode:bool,index_row:int)->Result:
        """
        Finalizamos la recogida de los resultados,
        Establecemos el uso y el exito del neurovector, así como los vectores y nodos
        """
         # Liberamos el bloqueo y preparamos la incorporación de una nueva fila al detalle
        self.blocked=False
        self.id_predict+=1
        self.detail.append([])
        final_result=None
        for var,workpace in self.outputs.items():
            result=workpace.cal_prediction(mode=mode,index_row=index_row)
            if result==None:
                break
            if result.success==None:
                raise ValueError(f"Analytics: No se ha podido establecer el éxito en variable {var}")
            if final_result==None:
                final_result=result 
            elif result.success:
                final_result=result

        return final_result
    
    def get_acuracy(self)->float:
        """
        Obtiene la media de la precisión de las variables
        """
        dic_accuracy=0
        dic_MAE=0
        dic_RMSE=0
        total_var=0
        header=['-','KPI','Variable','Pruebas','Éxitos','Precisión','RMSE','MAE']
        datos=[]

        if self.verbose:
            print (f"  COMPARADOR DE PRECISIÓN:")
        
        for var,data in self.outputs.items():
            if data.use>0:
                data.accuracy=data.success/data.use*100
                data.MAE=data.MAE/data.use
                data.RMSE=math.sqrt(data.RMSE/data.use)
            if self.verbose:
                print (f"RESULTADOS FINALES {var}: PRECISIÓN {data.accuracy:.2f}, MAE: {data.MAE:.2f}, RMSE: {data.RMSE:.2f}")
                
            dic_accuracy+=data.accuracy
            dic_MAE+=data.MAE
            dic_RMSE+=data.RMSE
            total_var+=1
            for name_method,method in data.methods.items():
                method.calculate()
                row=[]
                if name_method==self.sel_method:
                    row.append('*')
                else:
                    row.append(' ')
  
                row.extend([name_method,var,method.acc_use,method.acc_success,f"{method.accuracy*100:.2f}",f"{method.RMSE:.2f}",f"{method.MAE:.2f}"])
                datos.append(row)  

        if self.verbose:
            print(tabulate(datos, headers=header, tablefmt="pretty"))
        
        if total_var==0:
            if self.verbose:
                print ("No hay variables definidas para poder sacar la precisión")

            dic_accuracy=0
            dic_RMSE=0
            dic_MAE=0
        else:
            dic_MAE=dic_MAE/total_var
            dic_accuracy=dic_accuracy/total_var
            dic_RMSE=dic_RMSE/total_var
        
        return dic_accuracy,dic_MAE,dic_RMSE
    
    def get_results(self,sel_var:str='',sel_method:str='',curves:bool=False):
        if sel_var not in self.vars and sel_var!='':
            raise ValueError("La variable no está en la lista de variables")

        if sel_var=='':
            sel_var=self.vars
        else:
            sel_var=[sel_var]

        for var in sel_var:
            if curves:
                self.outputs[var].show_curves(sel_method)
            else:
                self.outputs[var].get_results(sel_method)

    def get_best_method(self,sel_var:str='',export_json:str=''):
        """
        Obtiene un listado con la tabla de competición de los algoritmos
        Permite esportar el mapa generado para luego poder usarlo
        """
        if sel_var not in self.vars and sel_var!='':
            raise ValueError("La variable no está en la lista de variables")

        if sel_var=='':
            sel_var=self.vars
        else:
            sel_var=[sel_var]
        dic_table={}
        for var in sel_var:
            dic_table[var],media_ponderada_final=self.outputs[var].runner_curves(verbose=self.verbose)
            if export_json!='':
                # Definir la variable adicional
                extra_variable = {"var": var}
                # Selección de las columnas deseadas
                df_to_export = dic_table[var][["Certeza (%)", "Mejor Metodo"]]
                # Combinar el DataFrame con la variable adicional
                combined_data = {
                    "map": df_to_export.to_dict(orient="records"),
                    **extra_variable
                }
                # Exportar a JSON solo las columnas seleccionadas
                with open(export_json, "w") as f:
                    json.dump(combined_data, f, indent=4)

        return dic_table


#END
