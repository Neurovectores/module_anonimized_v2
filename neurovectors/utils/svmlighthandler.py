# © 2025. Software anonimizado. Todos los derechos reservados.
# All rights reserved.
# Anónimo

import os
from io import BytesIO
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
import pandas as pd

class SVMLighthandler:
    """
    A handler class to convert SVMlight format files to CSV.
    
    This class provides methods to analyze SVMlight files, detect the maximum 
    number of features, and convert the data into a CSV format.
    """
    def __init__(self, file_path,new_file_path,chunk_size=10000,verbose=True):
        """
        Initializes the SVMLighthandler object with the given file paths and options.
        
        Args:
            file_path (str): The path to the SVMlight format file.
            new_file_path (str): The path where the converted CSV file will be saved.
            chunk_size (int): The number of lines to read at a time (default is 10000).
            verbose (bool): If True, a progress bar is displayed during header analysis.
        """
        self.verbose=verbose
        self.file_path=file_path
        self.new_file_path=new_file_path
        self.chunk_size=chunk_size
        self.total_size = os.path.getsize(file_path)
        self.max_features,self.headers = self.get_max_features()
        self.convert_svmlight_to_csv()

    
    def get_max_features(self):
        """
        Analyzes the SVMlight file to determine the maximum number of features and generates a header.
        
        The method reads the file in chunks, calculates the number of features for each chunk, 
        and updates the maximum number of features found. It also creates a header for the CSV file.

        Returns:
            max_features (int): The maximum number of features found in the file.
            header (list): The list of column names, starting with 'label' for the target variable.
        """
        max_features = 0
        processed_size = 0

        # Store the maximum number of features and the header
        header = []
        # Initialize progress bar if verbose is enabled
        if self.verbose:
            pbar = tqdm(total=self.total_size, unit='B', unit_scale=True, desc="Analyzing headers")

        with open(self.file_path, 'r') as file:
            while not self.cancel_flag:
                lines = []
                for _ in range(self.chunk_size):
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line)

                if not lines:
                    break

                # Create a BytesIO object from the lines
                buffer = BytesIO(''.join(lines).encode('utf-8'))
                X_chunk, _ = load_svmlight_file(buffer)
                n_features = X_chunk.shape[1]
                # Update the maximum number of features
                if n_features > max_features:
                    max_features = n_features
                    header = [f"col_{i}" for i in range(1, max_features + 1)]

                # Update the processed size for the progress bar
                processed_size += sum(len(line) for line in lines)
                if self.verbose:
                    pbar.update(processed_size)

        # Add 'label' as the first column in the header
        header.insert(0, 'label')
        # Close the progress bar
        if self.verbose:
            pbar.close()
        return max_features, header

    
    def convert_svmlight_to_csv(self):
        """
        Converts the SVMlight file to a CSV format.
        
        This method will load the SVMlight data in chunks and save it to a CSV file 
        with the appropriate headers.
        """
        processed_size = 0
        # Initialize progress bar if verbose is enabled
        if self.verbose:
            pbar = tqdm(total=self.total_size, unit='B', unit_scale=True, desc="Converting SVMlight to CSV")

        with open(self.new_file_path, 'w') as csv_file:
            header_written = False

            with open(self.file_path, 'r') as file:
                while not self.cancel_flag:
                    lines = []
                    for _ in range(self.chunk_size):
                        line = file.readline()
                        if not line:
                            break
                        lines.append(line)

                    if not lines:
                        break

                    # Create a BytesIO object from the lines
                    buffer = BytesIO(''.join(lines).encode('utf-8'))
                    processed_size = file.tell()
                    X_chunk, y_chunk = load_svmlight_file(buffer, n_features=self.max_features)
                    if X_chunk.shape[0] == 0:
                        break
        
                    # Convert the sparse matrix to COOrdinate format for easier manipulation
                    X_coo = X_chunk.tocoo()
                    # Create an empty DataFrame with the feature headers
                    df = pd.DataFrame(index=range(X_chunk.shape[0]), columns=self.headers)
                    df['label'] = y_chunk
                    # Fill the DataFrame only with the present values in the sparse matrix
                    for row, col, data in zip(X_coo.row, X_coo.col, X_coo.data):
                        df.iat[row, col + 1] = data  # col + 1 because the first column is 'label'

                    # Append to the CSV, writing the header only once
                    mode = 'w' if not header_written else 'a'
                    header = not header_written
                    header_written = True
                    
                    df.to_csv(self.new_file_path, mode=mode, header=header, index=False,lineterminator='\n')
                    # Update the progress bar
                    if self.verbose:
                        pbar.update(processed_size)
                    
        # Close the progress bar if it was opened
        if self.verbose:
            pbar.close()
