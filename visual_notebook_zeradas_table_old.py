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

class TableVisualization:
    """Visualização em formato de tabela dos resultados com DP."""
    
    def __init__(self, data=None, queries_config=None, data_path=None):
        """Initialize with provided data or from CSV."""
        try:
            # Create debug output widget first
            self.debug_output = widgets.Output()
            display(self.debug_output)
            
            with self.debug_output:
                print("Carregando App")
            
            # Setup basic configurations first
            self.hierarchy_levels = ['NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'CO_ENTIDADE']
            self.ordered_aggregations = ['QT_ALUNOS', 'MEDIA_NOTA', 'SOMA_NOTAS']
            
            if data is not None and queries_config is not None:
                # Use provided data
                self.df = data
                self.queries_config = queries_config
                with self.debug_output:
                    print(f"Using provided data. Shape: {self.df.shape}")
            
            elif data_path:
                # Load data from files
                self.data_path = Path(data_path)
                csv_path = self.data_path / "dp_results_stats_bq.csv"
                queries_file = self.data_path / "queries_formatadas_bq.csv"
                
                with self.debug_output:
                    print("Loading data...")
                    print(f"Reading CSV from: {data_path}")
                
                # Load data in chunks
                chunks = []
                chunk_size = 500000
                for chunk in pd.read_csv(
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
                ):
                    chunks.append(chunk)
                
                self.df = pd.concat(chunks, ignore_index=True)
                with self.debug_output:
                    print(f"Data loaded successfully. Shape: {self.df.shape}")

                # Load queries configuration
                self.queries_config = pd.read_csv(queries_file, sep=';')
            
            else:
                raise ValueError("Either data and queries_config or data_path must be provided")

            # Set aggregation options after data is loaded
            self.aggregation_options = [agg for agg in self.ordered_aggregations 
                                      if agg in self.df['aggregated_data'].str.upper().unique()]

            # Create widgets
            self._create_widgets()

            # Display interface
            self.display_interface()
            
        except Exception as e:
            with self.debug_output:
                print(f"Error in initialization: {str(e)}")
                print(traceback.format_exc())

    def _create_widgets(self):
        """Create the control widgets."""
        try:
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
            
            # Region, UF, Municipality dropdowns
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
            
            # Table output
            self.table_output = widgets.Output(
                layout=widgets.Layout(
                    height='800px',
                    width='100%',
                    border='1px solid #ddd',
                    overflow='auto'
                )
            )
            
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
        try:
            print("Connecting observers...")
            self.submit_button.on_click(self.update_table)
            
            # Add observer for hierarchy changes
            self.hierarchy_dropdown.observe(
                lambda change: print(f"Hierarchy changed to: {change.new}"), 
                names='value'
            )
            
            # Update geographic filters
            self.region_dropdown.observe(self._update_ufs, names='value')
            self.uf_dropdown.observe(self._update_municipios, names='value')
            
        except Exception as e:
            print(f"Error connecting observers: {str(e)}")
            print(traceback.format_exc())

    def update_table(self, button_clicked=None):
        """Update the table with current selections."""
        try:
            # Get current selections
            aggregation = self.aggregation_dropdown.value
            hierarchy = self.hierarchy_dropdown.value
            region = self.region_dropdown.value
            uf = self.uf_dropdown.value
            mun = self.mun_dropdown.value

            # Base filters
            filtered_df = self.df[
                (self.df['aggregated_data'].str.upper() == aggregation.upper())
            ]

            # Set grouping based on hierarchy
            if hierarchy == 'NO_REGIAO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'SG_UF']
                grouping_col = 'parent_regiao'
                entity_col = 'Região'
                total_col = 'Total de UFs'
            elif hierarchy == 'SG_UF':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'NO_MUNICIPIO']
                grouping_col = 'parent_uf'
                entity_col = 'UF'
                total_col = 'Total de Municípios'
            elif hierarchy == 'NO_MUNICIPIO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                grouping_col = 'parent_municipio'
                entity_col = 'Município'
                total_col = 'Total de Escolas'
            else:  # CO_ENTIDADE
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                grouping_col = 'group_by_val1'
                entity_col = 'Escola'
                total_col = 'Total de Registros'

            # Apply geographic filters
            if region != 'Todas':
                filtered_df = filtered_df[filtered_df['parent_regiao'] == region]
            if uf != 'Todas':
                filtered_df = filtered_df[filtered_df['parent_uf'] == uf]
            if mun != 'Todas':
                filtered_df = filtered_df[filtered_df['parent_municipio'] == mun]

            # Create base table with entity counts
            base_table = filtered_df.groupby(grouping_col).agg({
                'group_by_val1': 'nunique',  # Count unique entities
                'original_value': 'median'    # Get median of original values
            }).reset_index()

            # Count lost values
            lost_counts = filtered_df[
                (filtered_df['dp_avg'] == 0.0) | 
                filtered_df['dp_avg'].isna() | 
                (filtered_df['lost'] > 0)
            ].groupby(grouping_col)['group_by_val1'].nunique().reset_index()
            lost_counts.columns = [grouping_col, 'lost_count']

            # Merge lost counts with base table
            base_table = base_table.merge(
                lost_counts, 
                on=grouping_col, 
                how='left'
            )

            # Fill NaN values in lost_count with 0
            base_table['lost_count'] = base_table['lost_count'].fillna(0)

            # Rename columns
            base_table.columns = [grouping_col, total_col, 'original_value', 'lost_count']

            # Calculate and format percentage
            lost_col = 'Valores Perdidos'
            median_col = 'Mediana Original'
            base_table[lost_col] = base_table.apply(
                lambda row: f"{int(row['lost_count'])} ({(row['lost_count']/row[total_col]*100):.1f}%)" 
                if pd.notna(row['lost_count']) else "0 (0.0%)",
                axis=1
            )

            # Format median column
            base_table[median_col] = base_table['original_value'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "-"
            )

            # Select and order final columns
            base_table = base_table[[grouping_col, total_col, lost_col, median_col]]
            base_table = base_table.sort_values(total_col, ascending=False)

            # Display the table
            with self.table_output:
                clear_output()
                display(base_table)

        except Exception as e:
            print(f"Error updating table: {str(e)}")
            print(traceback.format_exc())

    def display_interface(self):
        """Display the interface components."""
        try:
            #with self.debug_output:
            #      print("Starting display_interface")
 
            # Create containers for each section
            title_section = widgets.VBox([
                widgets.HTML("<h2>Visualização de Valores Perdidos - Tabela</h2>")
            ])
            
            #with self.debug_output:
            #      print("Created title section")
            
            query_section = widgets.VBox([
                widgets.HTML("<b>Configuração da Query</b>"),
                widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
            ])
            
            #with self.debug_output:
            #      print("Created query section")
            

            filters_section = widgets.VBox([
                widgets.HTML("<b>Filtros Geográficos</b>"),
                widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
            ])

            #with self.debug_output:
            #      print("Created filters section")
            

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
            
            #with self.debug_output:
            #      print("Created main container, displaying...")

            # Display the main container
            display(main_container)
            
            with self.debug_output:
                print("Loading regions...")
            # Initialize geographic filters
            self._load_regions()
            
            #with self.debug_output:
            #      print("Connecting observers...")
            
            # Connect observers
            self._connect_observers()
            

            #with self.debug_output:
            #      print("Display interface complete")

        except Exception as e:
            with self.debug_output:
                print(f"Error displaying interface: {str(e)}")
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

