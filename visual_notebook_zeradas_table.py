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
        """Initialize the visualization."""
        print("Carregando App")
        
        # Load data
        print("Loading data...")
        self.load_data(data_path)
        
        # Create widgets
        print("Creating widgets:")
        self._create_widgets()
        
        # Display interface
        self.display_interface()
        
        # Update table with initial values
        self.update_table()

    def load_data(self, data_path):
        """Load data from CSV files."""
        try:
            self.data_path = Path(data_path)
            csv_path = self.data_path / "dp_results_stats_bq.csv"
            queries_file = self.data_path / "queries_formatadas_bq.csv"
            
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
            
        except Exception as e:
            print(f"Error in loading data: {str(e)}")
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
        try:
            print("Creating widgets:")
            
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
            '''
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
            '''
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
            
            print("- Creating table output...")
            self.table_output = widgets.Output(
                layout=widgets.Layout(
                    height='800px',
                    width='100%',
                    border='1px solid #ddd',
                    overflow='auto'
                )
            )
            
            print("All widgets created successfully!")
            
        except Exception as e:
            print(f"Error creating widgets: {str(e)}")
            print(traceback.format_exc())

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
        
        filters_controls = widgets.VBox([
            widgets.HTML("<b>Filtros Geográficos</b>"),
            widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
        ])
        
        return widgets.VBox([
            title,
            query_controls,
            filters_controls,
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

        # Update geographic filters
        self.region_dropdown.observe(self._update_ufs, names='value')
        self.uf_dropdown.observe(self._update_municipios, names='value')


    def update_table(self, button_clicked=None):
        """Update the table with current selections."""
        try:
            # Get current selections
            aggregation = self.aggregation_dropdown.value
            hierarchy = self.hierarchy_dropdown.value
            region = self.region_dropdown.value
            uf = self.uf_dropdown.value
            mun = self.mun_dropdown.value
            
            # Base filter for aggregation
            filtered_df = self.df[
                (self.df['aggregated_data'].str.upper() == aggregation.upper())
            ]
            
            # Apply geographic filters based on hierarchy
            if hierarchy == 'NO_REGIAO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'SG_UF']
                group_col = 'parent_regiao'
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]

                table_header = 'Total de UFs'
                groupby_entity = 'Região'
                
            elif hierarchy == 'SG_UF':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'NO_MUNICIPIO']
                group_col = 'parent_uf'
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]
                if uf != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == uf]
                
                table_header = 'Total de Municípios'
                groupby_entity = 'UF'
                
            elif hierarchy == 'NO_MUNICIPIO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                group_col = 'parent_municipio'
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]
                if uf != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == uf]
                if mun != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_municipio'] == mun]

                table_header = 'Total de Escolas'
                groupby_entity = 'Município' 
                               
            elif hierarchy == 'CO_ENTIDADE':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                group_col = 'parent_municipio'
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]
                if uf != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == uf]
                if mun != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_municipio'] == mun]

                table_header = 'Total de Escolas'
                groupby_entity = 'Município'
            
            # Get unique epsilon-delta combinations, ordered
            eps_delta_combinations = filtered_df[['epsilon', 'delta']].drop_duplicates().sort_values(
                ['epsilon', 'delta'],
                ascending=[True, True]
            )
            
            # Create base table with total entities
            base_table = filtered_df.groupby(group_col)['group_by_val1'].nunique().reset_index()
            base_table.columns = [group_col, table_header]
            
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
                    'group_by_val1': lambda g: g[
                        (combo_df['dp_avg'] == 0.0) | 
                        combo_df['dp_avg'].isna() | 
                        (combo_df['lost'] > 0)
                    ].nunique()
                }).reset_index()
                
                # Add to base table
                col_name = f'{table_header.split(" ")[1]} Perdidos (ε={eps}, δ={delta})'
                base_table = base_table.merge(
                    lost_counts,
                    on=group_col,
                    how='left'
                )
                base_table = base_table.rename(columns={'group_by_val1': col_name})
            
            # Rename the entity column for display
            base_table = base_table.rename(columns={group_col: groupby_entity})
            
            # Sort by Total Entidades in descending order
            base_table = base_table.sort_values(table_header, ascending=False)
            
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
            
            filters_section = widgets.VBox([
                widgets.HTML("<b>Filtros Geográficos</b>"),
                widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
            ])


            
            # Create main container
            main_container = widgets.VBox([
                title_section,
                query_section,
                filters_section,
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
            
            # Initialize geographic filters
            self._load_regions()
            
            # Connect observers
            self._connect_observers()
            
        except Exception as e:
            print(f"Error displaying interface: {str(e)}")
            print(traceback.format_exc())

    def _update_segmentation_options(self, change):
        """Update segmentation options based on aggregation selection."""
        try:
            # Get current aggregation
            aggregation = change.new
            
            # Filter queries for this aggregation
            agg_queries = self.queries_config[
                self.queries_config['aggregated_data'].str.upper() == aggregation.upper()
            ]
            
            # Get unique segmentation values
            seg2_options = set()
            seg3_options = set()
            
            for group_by in agg_queries['group_by'].dropna():
                cols = [col.strip() for col in group_by.split(',')]
                if len(cols) > 1:
                    seg2_options.add(cols[1])
                if len(cols) > 2:
                    seg3_options.add(cols[2])
            
            # Update dropdown options
            self.segment2_dropdown.options = sorted(seg2_options | {'Todas'})
            self.segment3_dropdown.options = sorted(seg3_options | {'Todas'})
            
        except Exception as e:
            print(f"Error updating segmentation options: {str(e)}")
            print(traceback.format_exc()) 

    def _update_ufs(self, change):
        """Handler para mudanças na região"""
        try:
            if change.new == 'Todas':
                self.uf_dropdown.options = ['Todas']
            else:
                # Debug print the data for this region
                mask = self.df['parent_regiao'].str.upper() == change.new.upper()
                
                # Get UFs for selected region from dataframe
                ufs = ['Todas'] + sorted([uf for uf in self.df[mask]['parent_uf'].unique() if pd.notna(uf)])
                
                self.uf_dropdown.options = ufs
            
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            print(f"Error in region change handler: {str(e)}")
            print(traceback.format_exc())

    def _update_municipios(self, change):
        """Handler para mudanças na UF"""
        try:
            if change.new == 'Todas':
                self.mun_dropdown.options = ['Todas']
            else:
                # Debug print the data for this UF
                mask = (
                    (self.df['parent_regiao'].str.upper() == self.region_dropdown.value.upper()) &
                    (self.df['parent_uf'].str.upper() == change.new.upper())
                )
                
                # Get municipalities for selected UF from dataframe
                municipios = ['Todas'] + sorted([m for m in self.df[mask]['parent_municipio'].unique() if pd.notna(m)])
                
                self.mun_dropdown.options = municipios
            
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            print(f"Error in UF change handler: {str(e)}")
            print(traceback.format_exc())

    def _on_mun_change(self, change):
        """Handler para mudanças no município"""
        try:
            print(f"\nMunicípio alterado para: {change.new}")
        except Exception as e:
            print(f"Error in municipality change handler: {str(e)}")
            print(traceback.format_exc())

    def _load_regions(self):
        """Load initial list of regions from reference CSV"""
        try:
            # Get unique regions from the main dataframe, handling NaN values
            regions = ['Todas'] + sorted([r for r in self.df['parent_regiao'].unique() if pd.notna(r)])
            
            # Update region dropdown
            self.region_dropdown.options = regions
            self.region_dropdown.value = 'Todas'  # Set initial value
            
            # Initialize UF and Municipality dropdowns
            self.uf_dropdown.options = ['Todas']
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            print(f"Error loading regions: {str(e)}")
            print(traceback.format_exc())

