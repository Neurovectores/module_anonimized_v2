# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo

from ..utils import CompareModel 
from . import Engine,Analytics,Sample
from .dataset import Dataset
import pandas as pd 
import matplotlib.pyplot as plt

class Actions:
    def __init__(self,proyecto):
        """
        Inicializa el modelo Neurovectors con un objeto Dataset.

        """
        self.verbose = proyecto.verbose
        self.seed = proyecto.seed

    def convert(self, dataset:Dataset,engine:Engine,row_start:int=0,rows:int=0,filter_rows:Sample=None):
        """
        Convierte un dataset en neurovectores.
        """                    
        total_rows=dataset.process(action=engine.conver_row,title=f"Convirtiendo a Neurovectores",row_start=row_start,rows= rows,filter_rows=filter_rows)
        if self.verbose:
            print (f"Convertidas {total_rows} filas")
        return total_rows    

    def train(self, dataset:Dataset,engine:Engine,analytics:Analytics, row_start:int=0,rows:int=0,
              scope:int=1,filter_rows:list=[],nan_value=None,epochs:int=1,stop:bool=True,mode:str='success'):
        """
        Entrena el modelo Neurovectors usando el dataset proporcionado.
        mode: indica de que forma vamos a entrenar:
            'success' = El modo por defecto solo aprende si no hay exito
            'all'     = Aprende todo, haya tenido exito o no. Es una conversión combinada
            'group'   = Intenta agrupar los nodos en los neurovectores
        """
        if not dataset.ready_train:
            raise ValueError("Dataset no preparado para el entrenamiento")
        
        # Ponemos las variables a estudiar
        analytics.set_vars(list_vars=dataset.get_study())
        analytics.scope=scope
        if self.verbose:
            print (" ")
            print (f"INICIO ENTRENAMIENTO {dataset.name} CON {engine.name} Y {epochs} ÉPOCAS, VARIABLES: {analytics.vars}")
            print ("--------------------------------------------------------------")
        for epoch in range(epochs):
            nvs=0
            # Creamos el espacio de trabajo del analytics
            analytics.new_analytics()
            if self.verbose:
                print (f"Epoca {epoch+1}/{epochs}--->")

            def test_row(index,row):
                nonlocal nvs  # Indicamos que nvs pertenece al ámbito de la función train
                result = engine.predict(analytics,stimulus=row,scope=scope,index_row=index)
                # Sino hemos acertado tenemos que convertir la fila
                if result==None:
                    raise ValueError("Entrenamiento: El Analytics no ha podido completar el resultado")
                
                if not result.success:
                    engine.conver_row(index,row,epoch=epoch)
                    nvs+=1
                elif result.success:
                    # El residuo lo añadimos al neurovector usando los datos del resultado del motor dependiendo de la opción
                    if mode=='all' and (result._candidate.knowledge!=1 or epoch==0):
                        engine.conver_row(index,row,epoch=epoch)
                        nvs+=1
                    elif mode=='group':
                        engine.add_nodes_to_nv(result=result)
                else:
                    raise ValueError("No podemos establecer el resultado")
                    


            total_rows=dataset.process(action=test_row,title='Entrenamiento ',row_start=row_start,rows=rows,filter_rows=filter_rows,nan=nan_value)
            if self.verbose:
                print (f"Total Neurovectores creados: {nvs} / {total_rows}")
            acuracy=analytics.get_acuracy()
            if acuracy==1 and stop:
                if self.verbose:
                    print(f"Se ha llegado al 100% de precisión en el entrnamiento en la epoca {epoch+1}")
                    break
        if self.verbose:
            print (" ")
            print ("FIN ENTRENAMIENTO")
            print (" ")
        return nvs

    def test(self, dataset:Dataset,engine:Engine,analytics:Analytics,row_start:int=0,rows:int=0,
             scope:int=1,filter_rows:Sample=None,nan_value=None,verbose:bool=None):
        """
        Prueba un dataset en neurovectores y busca los más parecidos mirando la precisión.
        Compara el primer nodo del vector y verifica si coincide con el vector más similar encontrado.
        """
        # Ponemos las variables a estudiar
        analytics.set_vars(list_vars=dataset.get_study())
        analytics.scope=scope
        if verbose==None:
            verbose=self.verbose
        
        if verbose:
            print (" ")
            print (f"INICIO TEST {dataset.name} CON {engine.name} Y FILTRO {filter_rows.name}")
            print ("--------------------------------------------------------------") 
        analytics.new_analytics()
        def test_row(index,row):
            result =engine.predict(analytics,row,scope=scope,mode=False,index_row=index)       
            if result is not None and not result.success:
                pass
                #analytics.show_predict()


        total_rows=dataset.process(action=test_row,title='Prueba del dataset',row_start=row_start,rows=rows,filter_rows=filter_rows,nan=nan_value)
        
        return analytics.get_acuracy()

    def predict(self, dataset:Dataset,engine:Engine,analytics:Analytics,row_start:int=0,rows:int=0,
             scope:int=1,filter_rows:Sample=None,nan_value=None):
        """
        Realiza predicciones usando el modelo entrenado, recoge los datos y opcionalmento lo exporta.

        """
        # Ponemos las variables a recolectar
        analytics.set_vars(list_vars=dataset.get_study())
        analytics.scope=scope
        if self.verbose:
            print (" ")
            print (f"INICIO TEST {dataset.name} CON {engine.name} Y FILTRO {filter_rows.name}")
            print ("--------------------------------------------------------------") 
        analytics.new_analytics()
        def test_row(index,row):
            result =engine.predict(analytics,row,scope=scope,mode=False,index_row=index)       
            if result is not None and not result.success:
                pass
                #analytics.show_predict()


        total_rows=dataset.process(action=test_row,title='Prueba del dataset',row_start=row_start,rows=rows,filter_rows=filter_rows,nan=nan_value)
        analytics.get_acuracy()
        
        return total_rows
 
    def show_graph(self,engine:Engine,nv_start:int=0,size:int=1):
        engine.show(nv_start,size)
   
    def compare_model(self,dataset:Dataset,train:Sample,test:Sample,nan_value:int=-1,seed:int=None,scale:bool=True,models:list=[]):
        """
        Ejecuta la comparación con varios algoritmos
        """
        dataset.reset_df()
        if seed==None:
            seed=self.seed
        model=CompareModel(dataset,train,test,seed=seed,nan_value=nan_value,scale=scale)
        all=(len(models)==0)
        if all or 'ramdom' in models:
            model.train_random_forest()
        if all or 'neural' in models:    
            model.train_neural_network()
        if all or 'svc' in models:
            model.train_svc()
        if all or 'gradient' in models:
            model.train_gradient_boosting()
    
    def varinfluence(self, dataset:Dataset,engine:Engine,analytics:Analytics,row_start:int=0,rows:int=0,
             scope:int=1,filter_rows:Sample=None,nan_value=None,influence_study=0):
        """
        Analiza la influencia de cada variable en la precisión final eliminándolas una a una.
        Muestra el impacto en Accuracy, MAE y RMSE.
        0=Accuracy, 1=MAE, 2=RMSE como factor de influencia
        """
        if self.verbose:
            print("\nCalculando baseline con todas las variables...")
        
        verbose_analytics=analytics.verbose
        
        start_accuracy=self.test(dataset=dataset,engine=engine,analytics=analytics,row_start=row_start,
                                rows=rows,scope=scope,filter_rows=filter_rows,nan_value=nan_value,verbose=False)
        # Cogemos el mapa certezas
        map_certainty=analytics.get_best_method()
        analytics.set_methods_map(map_certainty)
        # Volvemos a hacer el test con el mapa aplicado.
        print ("--> APLICANDO MAPA DE CERTEZAS")
        start_accuracy=self.test(dataset=dataset,engine=engine,analytics=analytics,row_start=row_start,
                        rows=rows,scope=scope,filter_rows=filter_rows,nan_value=nan_value,verbose=False)
        analytics.verbose=False
        # Elegimos la métrica base inicial
        baseline_value = start_accuracy[influence_study]
        # Este next_objective nos servirá para decidir la exclusión cuando haya mejora,
        # pero además conservamos baseline_value para calcular el delta.
        next_objective = baseline_value

        if self.verbose:
            print(f"--> Métrica base: {baseline_value} (influence_study={influence_study})")
            print("-------------------------")

        # Inicializamos las variables
        results_df = pd.DataFrame(columns=["pasada", "variable", "delta","accuracy"])  # Nuevo DataFrame para recoger los deltas
        results_best    =[]
        n_pass:int      = 0
        control:bool    = True
        limit_pass:int  = len(dataset.get_vars())
        best_var:str   = ''
        # Iniciamos el bucle par analizar que variables son las que quitandolas mejora el rendimiento
        while control:
            best_var=''
            list_vars = dataset.get_vars()
            if self.verbose:
                print(f"\n=== Pasada {n_pass+1} ===")
            
            # Iniciamos el bucle por todas las variables
            for i, var in enumerate(list_vars, start=1):
                if self.verbose:
                    print(f"Probando sin la variable: {var} ({i}/{len(list_vars)})")
                # Excluimos temporalmente la variable
                dataset.columns_info[var].exclude=True
                
                par_accuracy=self.test(dataset=dataset,engine=engine,analytics=analytics,row_start=row_start,
                                    rows=rows,scope=scope,filter_rows=filter_rows,nan_value=nan_value,verbose=False)
                # Si es precisión tiene que ser mayor
                current_metric = par_accuracy[influence_study]

                # Guardar la diferencia respecto al baseline_value
                # Si es Accuracy, la variación delta = (current_metric - baseline_value)
                # Si es MAE o RMSE, delta = (baseline_value - current_metric) si deseas ver "positivo" cuando mejora
                if influence_study==0:
                    # A mayor Accuracy, mejor. Así que delta>0 indica mejora
                    delta = current_metric - next_objective
                    # Accuracy mayor -> mejora
                    improved = (current_metric > next_objective)

                # Si es error tiene que ser menor
                else:
                    # A menor MAE o RMSE, mejor. Así que delta>0 indica mejora (disminución)
                    # => baseline - current_metric
                    delta = next_objective - current_metric
                    # Error menor -> mejora
                    improved = (current_metric < next_objective)
                # Registramos en el DataFrame
                results_df.loc[len(results_df)] = [n_pass+1, var, delta, current_metric]
                if improved:
                    best_var = var
                    next_objective = current_metric

                    if self.verbose:
                        if influence_study == 0:
                            print(f"\nMejora de Accuracy excluyendo {var}: {current_metric}")
                        else:
                            print(f"\nMejora de la métrica (MENOR es mejor) excluyendo {var}: {current_metric}")

                else:
                    if self.verbose:
                        print(f"No hay mejora con {var}, delta={delta: .4f}")


                dataset.columns_info[var].exclude=False
            
            # Finalizamos la recogida de muestras de la pasada

            if best_var=='' or n_pass>limit_pass:
                # Sino hemos econtrado mejora paramos el bucle o hemos hecho muchos bucles
                control=False
                if self.verbose:
                    print (f"Finalizamos, hemos hecho un total de {n_pass+1} pasadas")
            
            else:
                if self.verbose:
                    print ("\n")
                    print ("-----------------------------------------------------------------")
                    print (f"Hemos finalizado la pasada y excluimos definitivamente {best_var}")
                    print ("-----------------------------------------------------------------")
                dataset.columns_info[best_var].exclude=True
                results_best.append({best_var: next_objective})
            
            n_pass+=1      
        # Agrupar por variable y calcular la media del delta
        df_influence = results_df.groupby('variable')['delta'].mean().sort_values()  
        if self.verbose:
            print("\n============ RESUMEN DEL ANÁLISIS ============")
            print(f"Mejoras definitivas encontradas: {results_best}")
            print("\nDataFrame de resultados (delta respecto a baseline):")
            print(df_influence) 
        
        results_df.to_csv("ajustador.csv", index=False, encoding="utf-8")
        analytics.verbose=verbose_analytics  


        # Crear la figura y un eje (axes) para dibujar
        fig, ax = plt.subplots(figsize=(10, 6))

        # Dibujar un gráfico de barras con df_influence
        df_influence.plot(
            kind='bar',
            ax=ax,
            color='skyblue',
            edgecolor='black'
        )

        # Ajustar títulos y etiquetas
        ax.set_title('Media de influencia por variable (delta medio)', fontsize=14)
        ax.set_xlabel('Variables', fontsize=12)
        ax.set_ylabel('Delta medio respecto a la métrica base', fontsize=12)

        # Rotar las etiquetas del eje X para que no se corten
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Ajustar márgenes para que las etiquetas se vean completas
        plt.tight_layout()

        plt.show()
        
        return results_df


# END
