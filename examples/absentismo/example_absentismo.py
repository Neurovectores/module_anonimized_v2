# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved. 
#
# This software is provided solely for non-commercial use.   
# Refer to the LICENSE file for full terms and conditions.               
"""
Programa de ejemplo de cómo usar el paquete neurovectors con un dataset.
""" 
import sys
import os

# Añadir dos niveles superiores al PATH
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ruta_raiz)
from neurovectors import Proyecto

# Path to your CSV file
csv_absentismo_orig  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Absenteeism_at_work.csv'))
csv_absentismo_desc  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Absenteeism_at_work_desc.csv')) 


"""
# Parámetros generales
"""
nan_value = None  # Valor para datos vacíos en el proyecto
sel_method_train = 'Avg'
sel_method_test = 'Avg'

# Creación del proyecto
proyecto = Proyecto(titulo='Evaluación dataset Absentismo')

# Datasets: original y descompuesto
absentismo_original = proyecto.add_dataset(
	name='Absentismo original',
	file=csv_absentismo_orig,
	vars=['Absenteeism time in hours']
)
absentismo_original.set_exclude(['ID'])

absentismo_descompuesto = proyecto.add_dataset(
	name='Absentismo descompuesto',
	file=csv_absentismo_desc,
	vars=['_Absenteeism time in hours']
)

# Particiones aleatorias
test, train = proyecto.add_split('test', 'train', absentismo_original.n_rows, 20, seed=123)

# Motor y análisis
engine = proyecto.add_engine('Motor Absentismo')
analisis_train = proyecto.add_analytics('Clasificación Entrenamiento', method=sel_method_train)
analisis_test = proyecto.add_analytics('Clasificación Test', method=sel_method_test)

# Entrenamiento
nv = proyecto.actions.train(
	dataset=absentismo_descompuesto,
	engine=engine,
	analytics=analisis_train,
	row_start=0,
	rows=0,
	scope=2,
	filter_rows=train,
	nan_value=nan_value,
	epochs=5,
	stop=False,
	mode='all'
)

# Test inicial
proyecto.actions.test(
	dataset=absentismo_descompuesto,
	engine=engine,
	analytics=analisis_test,
	row_start=0,
	rows=0,
	scope=30,
	filter_rows=test,
	nan_value=nan_value
)

# Opcional: afinar métodos y repetir test
# analisis_test.set_methods_map(import_json='map_absentismo.json')
# analisis_train.methods_map = analisis_test.methods_map
# map_certainty = analisis_test.get_best_method()
# analisis_test.methods_map = map_certainty

# Comparación de modelos (escala=True recomendado para original)
print('Comparación escalando los datos')
proyecto.actions.compare_model(absentismo_original, train, test, scale=True, nan_value=0)






# analisis_test.get_results()
# analisis_test.get_results(curves=True)

# analisis_test.show_analytics()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

# engine.show_vectors()
# print ("Comparación sin escalar los datos")
# proyecto.actions.compare_model(absentismo_original,train,test,scale=False,nan_value=0)
 
# print ("Comparación con el dataset descompuesto y sin escalar")
# proyecto.actions.compare_model(absentismo_descompuesto,train,test,scale=False,nan_value=0)
# print ("Comparación con el dataset descompuesto y escalado")
# proyecto.actions.compare_model(absentismo_descompuesto,train,test,scale=True,nan_value=0)

# proyecto.show_project()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

# engine.save('absentismo_converter.nvs')
# engine.load('absentismo_converter.nvs')


#END



