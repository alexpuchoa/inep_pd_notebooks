import os
os.environ["PYTHONIOENCODING"] = "cp1252"

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import traceback
import logging

# Set the default renderer for Plotly to work in Colab
pio.renderers.default = 'colab'

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
            print("Carregando App")
            
            # Load data first
            self.data_path = Path(data_path)
            csv_path = self.data_path / "dp_results_stats_bq.csv"
            queries_file = self.data_path / "queries_formatadas_bq.csv"
            
            print("Loading data...")
            print(f"Reading CSV from: {data_path}")
            
            # Load data in chunks
            chunks = []
            chunk_size = 500000
            
            for i, chunk in enumerate(pd.read_csv(
                csv_path,
                dtype={
                    'epsilon': 'float64',
                    'delta': 'float64',
                    'dp_avg': 'float64',
                    'original_value': 'float64'
                },
                sep=';',
                encoding='latin1',
                low_memory=False,
                chunksize=chunk_size
            )):
                chunks.append(chunk)
            
            print("Concatenating chunks...")
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
            # Load queries configuration
            self.queries_config = pd.read_csv(queries_file, sep=';')
            print(f"Queries loaded. Shape: {self.queries_config.shape}")
            
            # Setup basic configurations
            self.hierarchy_levels = ['NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'CO_ENTIDADE']
            self.ordered_aggregations = ['QT_ALUNOS', 'MEDIA_NOTA', 'SOMA_NOTAS']
            self.aggregation_options = [agg for agg in self.ordered_aggregations 
                                      if agg in self.df['aggregated_data'].str.upper().unique()]
            
            # Create widgets
            print("\nCreating widgets:")
            self._create_widgets()
            
            # Display interface
            self.display_interface()
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
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
        
        return widgets.VBox([
            title,
            query_controls,
            segmentation_controls,
            self.submit_button
        ], layout=widgets.Layout(
            padding='20px',
            width='100%',
            border='1px solid #ddd',
            margin='10px'
        ))

    def _connect_observers(self):
        """Connect widget observers."""
        self.submit_button.on_click(self.update_table)
        self.aggregation_dropdown.observe(self._update_segmentation_options, names='value')

    def update_table(self, button_clicked=None):
        """Update the table with current selections."""
        try:
            # Get current selections
            aggregation = self.aggregation_dropdown.value
            hierarchy = self.hierarchy_dropdown.value
            seg2 = self.segment2_dropdown.value
            seg3 = self.segment3_dropdown.value
            
            # Base filter for aggregation
            filtered_df = self.df[
                (self.df['aggregated_data'].str.upper() == aggregation.upper())
            ]
            
            # Apply geographic filters based on hierarchy
            if hierarchy == 'NO_REGIAO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'SG_UF']
                group_col = 'parent_regiao'
                
            elif hierarchy == 'SG_UF':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'NO_MUNICIPIO']
                group_col = 'parent_uf'
                if seg2 != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == seg2]
                
            elif hierarchy == 'NO_MUNICIPIO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                group_col = 'parent_municipio'
                if seg2 != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == seg2]
                
            elif hierarchy == 'CO_ENTIDADE':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                group_col = 'parent_municipio'
                if seg2 != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_municipio'] == seg2]
            
            # Get unique epsilon-delta combinations
            eps_delta_combinations = filtered_df[['epsilon', 'delta']].drop_duplicates()
            
            # Create base table with total entities
            base_table = filtered_df.groupby(group_col)['group_by_val1'].nunique().reset_index()
            base_table.columns = [group_col, 'Total Entidades']
            
            # For each epsilon-delta combination, calculate lost entities
            for _, row in eps_delta_combinations.iterrows():
                eps, delta = row['epsilon'], row['delta']
                
                # Filter for this combination
                combo_df = filtered_df[
                    (filtered_df['epsilon'] == eps) & 
                    (filtered_df['delta'] == delta)
                ]
                
                # Count lost entities for this combination
                lost_counts = combo_df.groupby(group_col).agg({
                    'dp_avg': lambda x: ((x == 0.0) | x.isna() | (combo_df['lost'] > 0)).sum()
                }).reset_index()
                
                # Add to base table
                col_name = f'Perdidos (ε={eps}, δ={delta})'
                base_table = base_table.merge(
                    lost_counts,
                    on=group_col,
                    how='left'
                )
                base_table = base_table.rename(columns={'dp_avg': col_name})
            
            # Rename the entity column for display
            base_table = base_table.rename(columns={group_col: 'Entidade'})
            
            # Display table
            with self.table_output:
                clear_output(wait=True)
                display(HTML(base_table.to_html(
                    index=False,
                    float_format=lambda x: '{:.0f}'.format(x)
                )))
                print("\nTabela atualizada com sucesso!")
                
        except Exception as e:
            print(f"Error updating table: {str(e)}")
            print(traceback.format_exc())

    def display_interface(self):
        """Display the interface components."""
        try:
            # Create containers for each section
            title_section = widgets.VBox([
                widgets.HTML("<h2>Visualização de Valores Perdidos - Tabela</h2>")
            ])
            
            query_section = widgets.VBox([
                widgets.HTML("<b>Configuração da Query</b>"),
                widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
            ])
            
            segmentation_section = widgets.VBox([
                widgets.HTML("<b>Segmentações</b>"),
                widgets.HBox([self.segment2_dropdown, self.segment3_dropdown])
            ])
            
            # Create main container
            main_container = widgets.VBox([
                title_section,
                query_section,
                segmentation_section,
                self.submit_button,
                self.table_output
            ], layout=widgets.Layout(
                padding='20px',
                width='100%',
                border='1px solid #ddd',
                margin='10px'
            ))
            
            # Display the main container
            display(main_container)
            
            # Connect observers
            self._connect_observers()
            
        except Exception as e:
            print(f"Error displaying interface: {str(e)}")
            print(traceback.format_exc()) 