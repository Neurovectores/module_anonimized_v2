# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
#
# This software is provided solely for non-commercial use.   
# Refer to the LICENSE file for full terms and conditions.               
"""
Programa de ejemplo de cómo usar el paquete neurovectors con el dataset QSAR Biodegradation.
""" 
import sys
import os

# Añadir dos niveles superiores al PATH
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_raiz)
from neurovectors import Proyecto

# Path to your CSV file
csv_orig = os.path.abspath(os.path.join(os.path.dirname(__file__), 'biodeg.csv'))
csv_desc  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'biodeg_desc.csv'))

# Semilla
semilla=123

# Parámetros generales
nan_value       =   0    # Valor para datos vacíos en el proyecto
sel_method_test      =   'Avg' # Podemos usar cualquier método definido
sel_method_train     =   'Hits'

# Creación del proyecto
proyecto=Proyecto(titulo='Biodegradación basada en características moleculares')
dataset_original  =proyecto.add_dataset(name='Biodegradación original',file=csv_orig,vars=['class'])
# Si quisieramos descomponer el dataset original a digitaos
#dataset_original.to_digits()
#quit()
dataset_desc    =proyecto.add_dataset(name='Biodegradación descompuesto en dígitos',file=csv_desc,vars=['_class'])

# Creamos las particiones aleatorias
test,train=proyecto.add_split('test','train',dataset_original.n_rows,20,seed=semilla)
validation,train2 =proyecto.add_split('validation','train2',train.rows,10,seed=semilla)
conversion,train3 = proyecto.add_split('Conversión','train3',train2.rows,1,seed=semilla)
# Como opción podemos exportar el dataset partido usando los trozos aleatorios generados, creará dos ficheros usando el sample
# vinos_alenux.export(filter_rows=test)

# Añadimos el motor
engine=proyecto.add_engine('Motor1')

# Añadimos los análisis
analisis_train=proyecto.add_analytics("Clasificación Entrenamiento",method=sel_method_train)
analisis_test=proyecto.add_analytics("Clasificación Test",method=sel_method_test)
# Podemos cambiar en cualquier momento el método de análisis
#analisis_train.sel_method ='Avg_predict'
#analisis_test.sel_method  ='Avg_predict'

# Convertimos el dataset a neurovectores sin entrenamiento
# nv=proyecto.actions.convert(dataset=dataset_original,engine=engine,filter_rows=conversion)
# Aplicamos la mejora en el test final
#analisis_test.set_methods_map(import_json='map_methods.json')
#analisis_train.methods_map=analisis_test.methods_map
# Entrenamos indicando como queremos que sea el entrenamiento, las épocas, si queremos que pare si alcanza el 100% de precisión, etc.
nv=proyecto.actions.train(dataset=dataset_desc,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=2,filter_rows=train,nan_value=nan_value,epochs=5,stop=True,mode='all')
# nv=proyecto.actions.train(dataset=dataset_desc,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value,epochs=2,stop=False)
# Podemos pintar cualquier neurovector
#engine.paint_nv(nv=0)
# Con esto mostramos la información del proyecto
# proyecto.show_project()
# Muestra todos los datos del motor
# engine.show_all()

# Hacemos un test con los datos de validación
# proyecto.actions.test(dataset=dataset_desc,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value)
# Cogemos el mapa de certeza/ precisión de los métodos y lo establecemos en el analizador
#analisis_train.set_methods_map(map_certainty)
# Comparamos la mejora
# proyecto.actions.test(dataset=dataset_desc,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value)
# Entrenamos la validación para mejorar los resultados:
# nv=proyecto.actions.train(dataset=dataset_desc,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value,epochs=3)

# Iniciamos el test final
proyecto.actions.test(dataset=dataset_desc,engine=engine,analytics=analisis_test,row_start=0,rows=0,scope=30,filter_rows=test,nan_value=nan_value)
# Podemos exportar el mapa a un fichero json para usarlo luego
map_certainty=analisis_test.get_best_method()
analisis_test.set_methods_map(map_certainty)

proyecto.actions.test(dataset=dataset_desc,engine=engine,analytics=analisis_test,row_start=0,rows=0,scope=30,filter_rows=test,nan_value=nan_value)






#analisis_test.get_results()
#analisis_test.get_results(curves=True)

#analisis_test.show_analytics()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

#engine.show_vectors()
#print ("Comparación sin escalar los datos")
# proyecto.actions.compare_model(dataset_original,train,test,scale=False,nan_value=0)
print ("Comparación escalando los datos")
proyecto.actions.compare_model(dataset_original,train,test,scale=True,nan_value=0)
#print ("Comparación con el dataset descompuesto y sin escalar")
# proyecto.actions.compare_model(dataset_desc,train,test,scale=False)
#print ("Comparación con el dataset descompuesto y escalado")
# proyecto.actions.compare_model(dataset_desc,train,test,scale=True)
#print ("Comparación con el dataset descompuesto y escalado y nan a -1")
# proyecto.actions.compare_model(dataset_desc,train,test,scale=True,nan_value=-1)

# proyecto.show_project()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

#engine.save('vinos_converter.nvs')
#engine.load('vinos_converter.nvs')


#END



