import os
os.environ["PYTHONIOENCODING"] = "cp1252"

import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LostValuesTableVisualization:
    """Visualização tabular para contagem de valores perdidos."""
    
    def __init__(self, data_path):
        """Initialize the visualization interface with CSV data."""
        try:
            # Create and display logging output
            self.log_output = widgets.Output(
                layout=widgets.Layout(
                    border='1px solid #ddd',
                    padding='10px',
                    margin='10px 0'
                )
            )
            display(self.log_output)
            
            with self.log_output:
                print("Loading data...")
                print(f"Reading CSV from: {data_path}")
            
            # Load data
            self.data_path = Path(data_path)
            csv_path = self.data_path / "dp_results_stats_bq.csv"
            queries_file = self.data_path / "queries_formatadas_bq.csv"
            
            self.df = pd.read_csv(csv_path, 
                                dtype={
                                    'epsilon': 'float64',
                                    'delta': 'float64',
                                    'dp_avg': 'float64',
                                    'original_value': 'float64'
                                }, sep=';', encoding='latin1', low_memory=False)
            
            with self.log_output:
                print(f"Data loaded. Shape: {self.df.shape}")
            
            # Load queries configuration
            self.queries_config = pd.read_csv(queries_file, sep=';')
            
            # Define ordered lists
            self.hierarchy_levels = ['NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'CO_ENTIDADE']
            self.ordered_aggregations = ['QT_ALUNOS', 'MEDIA_NOTA', 'SOMA_NOTAS']
            
            # Extract unique aggregated_data values
            self.aggregation_options = [agg for agg in self.ordered_aggregations 
                                      if agg in self.df['aggregated_data'].str.upper().unique()]
            
            # Create segmentation mapping
            self._create_segmentation_map()
            
            # Create widgets
            self._create_widgets()
            
            # Initialize geographic filters
            self._load_regions()
            
            # Create and display container
            self.container = self._create_interface()
            display(self.container)
            
            # Create table output
            self.table_output = widgets.Output()
            display(self.table_output)
            
            # Connect observers
            self._connect_observers()
            
            with self.log_output:
                print("Initialization complete!")
            
        except Exception as e:
            with self.log_output:
                print(f"Error initializing visualization: {str(e)}")
                print(traceback.format_exc())
    
    def _create_segmentation_map(self):
        """Create mapping of segmentation options for each aggregation."""
        self.segmentation_map = {}
        for agg in self.aggregation_options:
            agg_queries = self.queries_config[self.queries_config['aggregated_data'].str.upper() == agg]
            group_by_values = agg_queries['group_by'].dropna().unique()
            
            seg2_options = set()
            seg3_options = set()
            
            for group_by in group_by_values:
                if pd.isna(group_by):
                    continue
                cols = [col.strip() for col in group_by.split(',')]
                if len(cols) > 1:
                    seg2_options.add(cols[1])
                if len(cols) > 2:
                    seg3_options.add(cols[2])
            
            self.segmentation_map[agg] = {
                'seg1': self.hierarchy_levels,
                'seg2': sorted(seg2_options | {'Todas'}),
                'seg3': sorted(seg3_options | {'Todas'})
            }
    
    def _create_widgets(self):
        """Create the control widgets."""
        # Aggregation type dropdown
        self.aggregation_dropdown = widgets.Dropdown(
            options=self.aggregation_options,
            value=self.aggregation_options[0],
            description='Agregação:'
        )
        
        # Hierarchical level dropdown
        self.hierarchy_dropdown = widgets.Dropdown(
            options=self.hierarchy_levels,
            value=self.hierarchy_levels[0],
            description='Nível:'
        )
        
        # Segmentation dropdowns
        self.segment2_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Segm. 2:'
        )
        
        self.segment3_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Segm. 3:'
        )
        
        # Geographic filters
        self.region_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Região:'
        )
        
        self.uf_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='UF:'
        )
        
        self.mun_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Município:'
        )
        
        # Update button
        self.submit_button = widgets.Button(
            description='Atualizar Tabela',
            button_style='primary',
            tooltip='Clique para atualizar a tabela com as seleções atuais'
        )
    
    def _create_interface(self):
        """Create the interface container."""
        title = widgets.HTML(
            "<h2 style='text-align: center; margin: 20px 0; color: #2c3e50;'>Visualização de Valores Perdidos - Tabela</h2>"
        )
        
        # Controls sections
        query_controls = widgets.VBox([
            widgets.HTML("<b>Configuração da Query</b>"),
            widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
        ])
        
        segmentation_controls = widgets.VBox([
            widgets.HTML("<b>Segmentações</b>"),
            widgets.HBox([self.segment2_dropdown, self.segment3_dropdown])
        ])
        
        geo_controls = widgets.VBox([
            widgets.HTML("<b>Filtros Geográficos</b>"),
            widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
        ])
        
        return widgets.VBox([
            title,
            query_controls,
            segmentation_controls,
            geo_controls,
            self.submit_button
        ], layout=widgets.Layout(
            padding='20px',
            width='100%',
            border='1px solid #ddd',
            margin='10px'
        )) 