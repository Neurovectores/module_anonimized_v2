# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
#
# This software is provided solely for non-commercial use. 
# Refer to the LICENSE file for full terms and conditions.  
#     
from __future__ import annotations
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np

class CompareModel:
    """
    Clase que comparar varios algoritmos predicitivos en base a un objeto Dataset
    Se usa un objeto Sample que contiene los datos de partición
    """
    def __init__(self, dataset, train:'Sample', test:'Sample',nan_value=-1,seed:int=123,scale:bool=True): # type: ignore
        from ..models import Sample
        self.dataset = dataset
        self.test_size = 0
        self.random_state = seed
        self.nan_value = nan_value
        self.sample_train=train
        self.sample_test=test
        self.df = self.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.scaler = StandardScaler()
        if scale:
            self.scale_data()

    def load_data(self):
        """
        Concatena el dataset para poder ser manipulado por los algoritmos predictivos,
        excluyendo las columnas marcadas previamente.
        """
        if not self.dataset.ready:
            raise ValueError("El dataset no está listo. Asegúrese de haber cargado los datos correctamente.")
        
        # Concatenar todos los chunks de una vez
        chunk = pd.concat(self.dataset.dataset_raw, ignore_index=True)
        
        # Obtener las columnas que se deben excluir
        excluded_columns = [
            col for col, info in self.dataset.columns_info.items()
            if getattr(info, 'exclude', False)
        ]
        
        # Eliminar columnas excluidas si están en el DataFrame
        chunk = chunk.drop(columns=[col for col in excluded_columns if col in chunk.columns])

        # Rellenar NaNs si se ha especificado un valor
        if self.nan_value is not None:
            chunk = chunk.fillna(self.nan_value)
        
        return chunk


    def split_data(self,var_pos:int=0):
        """
        Partimos los datos en las muestras pasadas
        var_pos: indican la posición de la variable de estudio en la lista de variables del dataset
        """
        # Obtenemos las variables de estudio
        vars=self.dataset.get_study()
        X = self.df.drop(columns=vars)
        y = self.df[vars[var_pos]]
        # Generar subconjuntos para entrenamiento y prueba en base a los índices
        X_train = X.iloc[self.sample_train.rows]
        y_train = y.iloc[self.sample_train.rows]
        X_test = X.iloc[self.sample_test.rows]
        y_test = y.iloc[self.sample_test.rows]
        return X_train,X_test,y_train,y_test

    def scale_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_random_forest(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf_model.fit(self.X_train, self.y_train)
        y_pred_rf = rf_model.predict(self.X_test)
        
        # Calcular métricas
        accuracy_rf = accuracy_score(self.y_test, y_pred_rf) * 100
        mae_rf = mean_absolute_error(self.y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(self.y_test, y_pred_rf))
        
        print(f"Modelo RandomForest - Precisión: {accuracy_rf:.2f}%, MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")
    
    def train_neural_network(self):
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=2000, random_state=self.random_state)
        nn_model.fit(self.X_train, self.y_train)
        y_pred_nn = nn_model.predict(self.X_test)
        
        # Calcular métricas
        accuracy_nn = accuracy_score(self.y_test, y_pred_nn) * 100
        mae_nn = mean_absolute_error(self.y_test, y_pred_nn)
        rmse_nn = np.sqrt(mean_squared_error(self.y_test, y_pred_nn))
        
        print(f"Modelo Red Neuronal - Precisión: {accuracy_nn:.2f}%, MAE: {mae_nn:.4f}, RMSE: {rmse_nn:.4f}")
    
    def train_svc(self):
        svc_model = SVC(kernel='rbf', random_state=self.random_state)
        svc_model.fit(self.X_train, self.y_train)
        y_pred_svc = svc_model.predict(self.X_test)
        
        # Calcular métricas
        accuracy_svc = accuracy_score(self.y_test, y_pred_svc) * 100
        mae_svc = mean_absolute_error(self.y_test, y_pred_svc)
        rmse_svc = np.sqrt(mean_squared_error(self.y_test, y_pred_svc))
        
        print(f"Modelo SVC - Precisión: {accuracy_svc:.2f}%, MAE: {mae_svc:.4f}, RMSE: {rmse_svc:.4f}")
    
    def train_gradient_boosting(self):
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        gb_model.fit(self.X_train, self.y_train)
        y_pred_gb = gb_model.predict(self.X_test)
        
        # Calcular métricas
        accuracy_gb = accuracy_score(self.y_test, y_pred_gb) * 100
        mae_gb = mean_absolute_error(self.y_test, y_pred_gb)
        rmse_gb = np.sqrt(mean_squared_error(self.y_test, y_pred_gb))
        
        print(f"Modelo Gradient Boosting - Precisión: {accuracy_gb:.2f}%, MAE: {mae_gb:.4f}, RMSE: {rmse_gb:.4f}")
