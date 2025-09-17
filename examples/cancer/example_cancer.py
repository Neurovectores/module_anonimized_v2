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
csv_cancer_orig  =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'breastcancer.csv'))
csv_cancer_desc  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'breastcancer_desc.csv'))
# El csv en vez de B y M con 1 y 0
csv_cancer_cat      = os.path.abspath(os.path.join(os.path.dirname(__file__), 'breastcancer_cat.csv'))
csv_cancer_desc_cat = os.path.abspath(os.path.join(os.path.dirname(__file__), 'breastcancer_desc_cat.csv'))

# Parámetros generales
nan_value            =   None    # Valor para datos vacíos en el proyecto
sel_method_test      =   'Hits' # Podemos usar cualquier método definido
sel_method_train     =   'Hits'

# Creación del proyecto
proyecto=Proyecto(titulo='Evaluación dataset Cáncer')
cancer_original =proyecto.add_dataset(name='Cáncer original',file=csv_cancer_orig,vars=['y'])
cancer_original_cat =proyecto.add_dataset(name='Cáncer original (cat)',file=csv_cancer_cat,vars=['y'])
# Si quisieramos descomponer el dataset original a digitaos
#cancer_original.to_digits()
cancer_descompueto=proyecto.add_dataset(name='Cáncer descompuesto',file=csv_cancer_desc,vars=['_y'])
cancer_descompueto_cat=proyecto.add_dataset(name='Cáncer descompuesto (cat)',file=csv_cancer_desc_cat,vars=['_y'])
# Creamos las particiones aleatorias
test,train=proyecto.add_split('test','train',cancer_original.n_rows,20,seed=123)
validation,train2 =proyecto.add_split('validation','train2',train.rows,10)
conversion,train3 = proyecto.add_split('Conversión','train3',train2.rows,1)

# Añadimos el motor
engine=proyecto.add_engine('Motor Cáncer')

# Añadimos los análisis
analisis_train=proyecto.add_analytics("Clasificación Entrenamiento",method=sel_method_train)
analisis_test=proyecto.add_analytics("Clasificación Test",method=sel_method_test)
# Podemos cambiar en cualquier momento el método de análisis
analisis_test.sel_method  ='Vectors_Energy'
#analisis_train.sel_method  ='Vectors_Energy'

# Convertimos el dataset a neurovectores sin entrenamiento
# nv=proyecto.actions.convert(dataset=cancer_descompuesto,engine=engine,filter_rows=conversion)
# Aplicamos la mejora en el test final
#analisis_test.set_methods_map(import_json='map_methods.json')
#analisis_train.methods_map=analisis_test.methods_map

# Entrenamos indicando como queremos que sea el entrenamiento, las épocas, si queremos que pare si alcanza el 100% de precisión, etc.
nv=proyecto.actions.train(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=train,nan_value=nan_value,epochs=5,stop=False,mode='all')
# nv=proyecto.actions.train(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value,epochs=2,stop=False)
# Podemos pintar cualquier neurovector
#engine.paint_nv(nv=0)
# Con esto mostramos la información del proyecto
# proyecto.show_project()
# Muestra todos los datos del motor
# engine.show_all()

# Hacemos un test con los datos de validación
# proyecto.actions.test(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value)
# Cogemos el mapa de certeza/ precisión de los métodos y lo establecemos en el analizador
#analisis_train.set_methods_map(map_certainty)
# Comparamos la mejora
# proyecto.actions.test(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value)
# Entrenamos la validación para mejorar los resultados:
# nv=proyecto.actions.train(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_train,row_start=0,rows=0,scope=10,filter_rows=validation,nan_value=nan_value,epochs=3)

# Iniciamos el test final
proyecto.actions.test(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_test,row_start=0,rows=0,scope=30,filter_rows=test,nan_value=nan_value)
# Podemos exportar el mapa a un fichero json para usarlo luego
map_certainty=analisis_test.get_best_method()


# proyecto.actions.test(dataset=cancer_descompueto_cat,engine=engine,analytics=analisis_test,row_start=0,rows=0,scope=30,filter_rows=test,nan_value=nan_value)






#analisis_test.get_results()
#analisis_test.get_results(curves=True)

#analisis_test.show_analytics()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

#engine.show_vectors()
print ("Comparación sin escalar los datos")
proyecto.actions.compare_model(cancer_original_cat,train,test,scale=False,nan_value=0)
print ("Comparación escalando los datos")
proyecto.actions.compare_model(cancer_original_cat,train,test,scale=True,nan_value=0)
print ("Comparación con el dataset descompuesto y sin escalar")
proyecto.actions.compare_model(cancer_descompueto_cat,train,test,scale=False,nan_value=0)
print ("Comparación con el dataset descompuesto y escalado")
proyecto.actions.compare_model(cancer_descompueto_cat,train,test,scale=True,nan_value=0)
#print ("Comparación con el dataset descompuesto y escalado y nan a -1")
# proyecto.actions.compare_model(cancer_descompueto_cat,train,test,scale=True,nan_value=-1)

# proyecto.show_project()

# proyecto.actions.show_graph(engine=engine,nv_start=0,size=1)

#engine.save('vinos_converter.nvs')
#engine.load('vinos_converter.nvs')


#END



