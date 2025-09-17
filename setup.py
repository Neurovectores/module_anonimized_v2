from setuptools import setup, find_packages

setup(
    name='neurovectors',  # Nombre del nuevo paquete
    version='2.0.1',
    packages=find_packages(),  # Encuentra automáticamente todas las subcarpetas con un __init__.py
    install_requires=[
        'pandas>=1.5.0',        # Añade pandas como dependencia
        'tqdm>=4.0.0',          # Añade tqdm como dependencia
        'scikit-learn>=0.24.0', # Añade scikit-learn como dependencia
        'Babel>=2.9.0',         # Añade babel como dependencia
        'psutil>=5.8.0',        # Añade psutil como dependencia
        'numpy>=1.21.0',        # Añade numpy como dependencia
        'tabulate>=0.8.9',      # Añade tabulate como dependencia
        'matplotlib>=3.4.0',    # Añade matplotlib como dependencia
        'pandasgui>=0.2.12',    # Añade pandasgui como dependencia
        'networkx>=2.6.0',      # Añade networkx como dependencia
        'pyvis>=0.3.2',         # Añade pyvis como dependencia
    ],  
    author='Anónimo',
    description='Paquete Neurovectors para algoritmos predictivos',
    long_description=open('README.md').read(),  # Puedes añadir un archivo README para la documentación
    long_description_content_type='text/markdown',  # Especifica el tipo de contenido (Markdown en este caso)
    license='Commercial License',  # Indica que es una licencia comercial
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
    ],
    python_requires='>=3.11',  # Indica las versiones de Python soportadas
)
