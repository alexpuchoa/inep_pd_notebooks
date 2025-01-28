import os
os.environ["PYTHONIOENCODING"] = "cp1252"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
import plotly.express as px
import traceback

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LostValuesVisualizationColab:
    """Visualização específica para contagem de valores perdidos - Versão Colab."""
    
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
                print(f"Chunk {i+1} loaded...")
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
            
            # Initialize geographic filters
            self._load_regions()
            
            # Display each component individually
            print("\nDisplaying interface components:")
            
            # Title
            display(widgets.HTML("<h2>Visualização de Valores Perdidos</h2>"))
            
            # Query controls
            display(widgets.VBox([
                widgets.HTML("<b>Configuração da Query</b>"),
                widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
            ]))
            
            # Segmentation controls
            display(widgets.VBox([
                widgets.HTML("<b>Segmentações</b>"),
                widgets.HBox([self.segment2_dropdown, self.segment3_dropdown])
            ]))
            
            # DP Parameters
            display(widgets.VBox([
                widgets.HTML("<b>Parâmetros DP</b>"),
                widgets.HBox([self.epsilon_dropdown, self.delta_dropdown])
            ]))
            
            # Geographic filters
            display(widgets.VBox([
                widgets.HTML("<b>Filtros Geográficos</b>"),
                widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
            ]))
            
            # Submit button
            display(self.submit_button)
            
            # Plots area
            display(widgets.HTML("<h3>Gráficos:</h3>"))
            display(self.plots_output)
            
            # Connect observers
            self._connect_observers()
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            print(traceback.format_exc())

    def _create_widgets(self):
        """Create the control widgets."""
        try:
            print("- Creating aggregation dropdown...")
            
            # Aggregation type dropdown
            self.aggregation_dropdown = widgets.Dropdown(
                options=self.aggregation_options,
                value=self.aggregation_options[0],
                description='Agregação:'
            )
            
            print("- Creating hierarchy dropdown...")
            # Hierarchical level dropdown
            self.hierarchy_dropdown = widgets.Dropdown(
                options=self.hierarchy_levels,
                value=self.hierarchy_levels[0],
                description='Nível:'
            )
            
            print("- Creating segmentation dropdowns...")
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
            
            print("- Creating geographic filters...")
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
            
            print("- Creating DP parameters...")
            # DP parameters
            self.epsilon_dropdown = widgets.Dropdown(
                options=sorted(self.df['epsilon'].unique()),
                description='Epsilon:'
            )
            
            self.delta_dropdown = widgets.Dropdown(
                options=sorted(self.df['delta'].unique()),
                description='Delta:'
            )
            
            print("- Creating submit button...")
            # Submit button
            self.submit_button = widgets.Button(
                description='Atualizar Gráfico',
                button_style='primary',
                tooltip='Clique para atualizar o gráfico'
            )
            
            print("- Creating figure widgets...")
            # Create output widget for plots
            self.plots_output = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    height='800px',
                    border='1px solid #ddd',
                    margin='20px 0'
                )
            )
            
            print("All widgets created successfully!")
            
        except Exception as e:
            print(f"Error creating widgets: {str(e)}")
            print(traceback.format_exc())

    def display_chart(self):
        """Create the visualization interface."""
        try:
            print("\nBuilding display:")
            
            # Create main interface container
            interface = widgets.VBox([
                widgets.HTML(
                    "<h2 style='text-align: center; margin: 20px 0; color: #2c3e50;'>Visualização de Valores Perdidos</h2>"
                ),
                widgets.HTML(
                    "<h4 style='text-align: left; margin: 15px 0; color: #2c3e50;'>NIVEL - Como os dados zerados do nivel hirárquico logo abaixo serão agregados.</h4>"
                ),
                widgets.VBox([
                    widgets.HTML("<b>Configuração da Query</b>"),
                    widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
                ]),
                widgets.VBox([
                    widgets.HTML("<b>Segmentações</b>"),
                    widgets.HBox([self.segment2_dropdown, self.segment3_dropdown])
                ]),
                widgets.VBox([
                    widgets.HTML("<b>Parâmetros DP</b>"),
                    widgets.HBox([self.epsilon_dropdown, self.delta_dropdown])
                ]),
                widgets.VBox([
                    widgets.HTML("<b>Filtros Geográficos</b>"),
                    widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
                ]),
                self.submit_button,
                widgets.HTML("<h3>Gráficos:</h3>"),
                self.plots_output
            ], layout=widgets.Layout(
                padding='20px',
                width='100%',
                border='1px solid #ddd',
                margin='10px'
            ))
            
            print("Display interface built successfully")
            return interface
            
        except Exception as e:
            print(f"Error creating interface: {str(e)}")
            print(traceback.format_exc())

    def _load_regions(self):
        """Load initial list of regions from reference CSV"""
        try:
            # Debug print the unique values in parent_regiao
            #self.debug_print("Available regions in dataframe:")
            #self.debug_print(self.df['parent_regiao'].unique())
            
            # Get unique regions from the main dataframe, handling NaN values
            regions = ['Todas'] + sorted([r for r in self.df['parent_regiao'].unique() if pd.notna(r)])
            #self.debug_print(f"Filtered regions: {regions}")
            
            # Update region dropdown
            self.region_dropdown.options = regions
            self.region_dropdown.value = 'Todas'  # Set initial value
            
            # Initialize UF and Municipality dropdowns
            self.uf_dropdown.options = ['Todas']
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'
            
            #self.debug_print("Regions loaded successfully")
            
        except Exception as e:
            self.debug_print(f"Error loading regions: {str(e)}")
            self.debug_print(traceback.format_exc())

    def update_plot(self, button_clicked=None):
        """Update plot with the current selections."""
        try:
            print("\nUpdating plots...")
            
            # Validate required selections based on hierarchy level
            hierarchy_level = self.hierarchy_dropdown.value
            
            if hierarchy_level == 'SG_UF' and self.region_dropdown.value == 'Todas':
                self.debug_print("Error: Uma região deve ser selecionada quando o nível é UF")
                return
                
            elif hierarchy_level == 'NO_MUNICIPIO':
                if self.region_dropdown.value == 'Todas':
                    self.debug_print("Error: Uma região deve ser selecionada quando o nível é Município")
                    return
                if self.uf_dropdown.value == 'Todas':
                    self.debug_print("Error: Uma UF deve ser selecionada quando o nível é Município")
                    return
                    
            elif hierarchy_level == 'CO_ENTIDADE':
                if self.region_dropdown.value == 'Todas':
                    self.debug_print("Error: Uma região deve ser selecionada quando o nível é Escola")
                    return
                if self.uf_dropdown.value == 'Todas':
                    self.debug_print("Error: Uma UF deve ser selecionada quando o nível é Escola")
                    return
                if self.mun_dropdown.value == 'Todas':
                    self.debug_print("Error: Um município deve ser selecionado quando o nível é Escola")
                    return
            '''
            self.debug_print("\n=== Starting Update ===")
            self.debug_print(f"Aggregation: {self.aggregation_dropdown.value}")
            self.debug_print(f"Hierarchy: {hierarchy_level}")
            self.debug_print(f"Epsilon: {self.epsilon_dropdown.value}")
            self.debug_print(f"Delta: {self.delta_dropdown.value}")
            self.debug_print(f"Region: {self.region_dropdown.value}")
            self.debug_print(f"UF: {self.uf_dropdown.value}")
            self.debug_print(f"Municipality: {self.mun_dropdown.value}")
            '''
            # Get results only if validations pass
            results = self.get_lost_values_results()
            
            if results is None or results.empty:
                self.debug_print("No results to plot")
                return
            
            #self.debug_print(f"Retrieved {len(results)} rows")
            self.plot_lost_values(results)
            
            # Create and display plots
            with self.plots_output:
                clear_output(wait=True)
                
                # Create percentage plot
                fig_percentages = go.Figure()
                fig_percentages.add_trace(go.Bar(
                    x=results['group_by_val1'],
                    y=results['lost_entities'] / results['total_entities'] * 100,
                    name='Percentual',
                    text=[f'{p:.1f}%' for p in results['lost_entities'] / results['total_entities'] * 100],
                    textposition='auto',
                ))
                fig_percentages.update_layout(
                    title='Percentual de Valores Perdidos',
                    xaxis_title='Valor',
                    yaxis_title='Percentual (%)',
                    showlegend=True,
                    height=400,
                    width=900,
                    margin=dict(t=50, b=50)
                )
                display(fig_percentages)
                
                # Create totals plot
                fig_totals = go.Figure()
                fig_totals.add_trace(go.Bar(
                    x=results['group_by_val1'],
                    y=results['total_entities'],
                    name='Total',
                    text=[f'Total: {total}' for total in results['total_entities']],
                    textposition='auto',
                ))
                fig_totals.update_layout(
                    title='Total de Valores Perdidos',
                    xaxis_title='Valor',
                    yaxis_title='Total',
                    showlegend=True,
                    height=400,
                    width=900,
                    margin=dict(t=50, b=50)
                )
                display(fig_totals)
            
            print("Plots updated successfully!")
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            print(traceback.format_exc())

    def debug_print(self, *messages):
        """Helper method to print debug messages. Can handle multiple arguments."""
        with self.debug_output:
            print(*messages)

    def get_lost_values_results(self):
        """
        Get lost values results using DataFrame operations instead of SQL.
        """
        try:
            hierarchy_level = self.hierarchy_dropdown.value
            
            # Base conditions present in all queries
            base_mask = (
                (self.df['aggregated_data'].str.upper() == self.aggregation_dropdown.value.upper()) &
                (self.df['epsilon'] == self.epsilon_dropdown.value) &
                (self.df['delta'] == self.delta_dropdown.value)
            )
            
            if hierarchy_level == 'NO_REGIAO':
                # Region level query
                mask = base_mask & (self.df['group_by_col1'].str.upper() == 'SG_UF')

                #self.debug_print(f"Current UF selection: {self.region_dropdown.value}")
                if self.region_dropdown.value != 'Todas':
                    #self.debug_print("A UF has been selected!")
                    mask = mask & (self.df['parent_regiao'] == self.region_dropdown.value)
                
                # Add segmentation filters if selected
                if self.segment2_dropdown.value != 'Todas':
                    mask = mask & (self.df['segmentation2'] == self.segment2_dropdown.value)
                if self.segment3_dropdown.value != 'Todas':
                    mask = mask & (self.df['segmentation3'] == self.segment3_dropdown.value)
                
                filtered_df = self.df[mask]
                group_col = 'parent_regiao'
                
            elif hierarchy_level == 'SG_UF':
                # State level query
                mask = base_mask & (
                    (self.df['parent_regiao'].str.upper() == self.region_dropdown.value.upper()) &
                    (self.df['group_by_col1'].str.upper() == 'NO_MUNICIPIO')
                )

                #self.debug_print(f"Current UF selection: {self.uf_dropdown.value}")
                if self.uf_dropdown.value != 'Todas':
                    #self.debug_print("A UF has been selected!")
                    mask = mask & (self.df['parent_uf'] == self.uf_dropdown.value)
                    
                filtered_df = self.df[mask]
                group_col = 'parent_uf'
                
            elif hierarchy_level == 'NO_MUNICIPIO':
                # Municipality level query
                mask = base_mask & (
                    (self.df['parent_regiao'].str.upper() == self.region_dropdown.value.upper()) &
                    (self.df['parent_uf'].str.upper() == self.uf_dropdown.value.upper()) &
                    (self.df['group_by_col1'].str.upper() == 'CO_ENTIDADE')
                )

                #self.debug_print(f"Current Municipality selection: {self.mun_dropdown.value}")
                if self.mun_dropdown.value != 'Todas':
                    #self.debug_print("A Municipality has been selected!")
                    mask = mask & (self.df['parent_municipio'] == self.mun_dropdown.value)
                    
                filtered_df = self.df[mask]
                group_col = 'parent_municipio'
                
            elif hierarchy_level == 'CO_ENTIDADE':
                # School level query
                mask = base_mask & (
                    (self.df['parent_regiao'].str.upper() == self.region_dropdown.value.upper()) &
                    (self.df['parent_uf'].str.upper() == self.uf_dropdown.value.upper()) &
                    (self.df['parent_municipio'].str.upper() == self.mun_dropdown.value.upper())
                )
                #self.debug_print(f"Current School selection: {self.mun_dropdown.value}")
                if self.mun_dropdown.value != 'Todas':
                    #self.debug_print("A Municipality has been selected!")
                    mask = mask & (self.df['parent_municipio'] == self.mun_dropdown.value)
                    
                filtered_df = self.df[mask]
                group_col = 'parent_municipio'
            
            else:
                self.debug_print(f"Invalid hierarchy level: {hierarchy_level}")
                return pd.DataFrame()
            
            # Common aggregation logic for all levels
            results = (filtered_df
                .groupby(group_col)
                .agg({
                    'group_by_val1': 'count',  # total_entities
                    'dp_avg': lambda x: ((x == 0.0) | x.isna()).sum(),  # lost_entities
                    'original_value': ['median', lambda x: x[((filtered_df['dp_avg'] == 0.0) | filtered_df['dp_avg'].isna())].median()]
                })
            )
            
            # Format results
            results = results.reset_index()
            results.columns = ['group_by_val1', 'total_entities', 'lost_entities', 'median_original', 'median_original_lost']
            
            # Debug information
            self.debug_print(f"Filtered data shape: {filtered_df.shape[0]}")
            self.debug_print(f"Results shape: {results.shape}")
            
            return results.sort_values('group_by_val1')
            
        except Exception as e:
            self.debug_print(f"Error in get_lost_values_results: {str(e)}")
            self.debug_print(traceback.format_exc())
            return pd.DataFrame(columns=['group_by_val1', 'lost_entities', 'total_entities', 
                                      'median_original', 'median_original_lost'])

    def plot_lost_values(self, results):
        """
        Plota dois gráficos separados usando VBox.
        """
        try:
            if results.empty:
                self.debug_print("No data to plot")
                return

            hierarchy_desc = {
                'NO_REGIAO': 'UFs',
                'SG_UF': 'Municípios',
                'NO_MUNICIPIO': 'Escolas',
                'CO_ENTIDADE': 'Escolas'
            }[self.hierarchy_dropdown.value]
                
            # Clear existing plots
            self.fig_percentages = go.Figure()
            self.fig_totals = go.Figure()
            
            # Calculate percentages
            percentages = [
                (lost/total * 100) if total > 0 else 0 
                for lost, total in zip(results['lost_entities'], results['total_entities'])
            ]
            
            # Update top figure - Scatter plot with percentages
            self.fig_percentages.add_trace(
                go.Scatter(
                    x=results['group_by_val1'],
                    y=percentages,
                    mode='markers+text',
                    marker=dict(
                        size=6,
                        color='rgb(55, 83, 109)',
                        symbol='circle'
                    ),
                    text=[f'{p:.1f}%' for p in percentages],
                    textposition='top center',
                    name='% Perdidos'
                )
            )
            
            # Update bottom figure - Bar chart with totals and lost entities
            self.fig_totals.add_trace(
                go.Bar(
                    x=results['group_by_val1'],
                    y=results['total_entities'],
                    name='Total',
                    marker_color='rgb(158,27,225)', # 158,202,225
                    text=[f'Total: {total}<br>Perdidos: {lost}' for total, lost in zip(results['total_entities'], results['lost_entities'])],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' +
                                'Total: %{y}<br>' +
                                'Perdidos: %{customdata}<br>' +
                                '<extra></extra>',
                    customdata=results['lost_entities']
                )
            )
            
            # Update layouts with adjusted dimensions
            self.fig_percentages.update_layout(
                height=500,  # Keep increased height
                width=1000,  # Reduced from 1200
                title=f'Percentual de {hierarchy_desc} com Valores Perdidos',
                yaxis=dict(
                    title='Percentual (%)',
                    range=[0, 100],
                    fixedrange=True
                ),
                xaxis=dict(tickangle=45),
                margin=dict(t=50, b=50, l=100, r=100)
            )
            
            self.fig_totals.update_layout(
                height=500,  # Keep increased height
                width=1000,  # Reduced from 1200
                title=f'Total de {hierarchy_desc} Existentes e com Valores Perdidos(as)',
                yaxis_title='Quantidade',
                xaxis=dict(tickangle=45),
                margin=dict(t=50, b=50, l=100, r=100)
            )
            
        except Exception as e:
            self.debug_print(f"Error plotting: {str(e)}")
            self.debug_print(traceback.format_exc())

    def _get_plot_title(self):
        """Helper method to generate plot title"""
        hierarchy_desc = {
            'NO_REGIAO': 'Regiões',
            'SG_UF': 'UFs',
            'NO_MUNICIPIO': 'Municípios',
            'CO_ENTIDADE': 'Escolas'
        }[self.hierarchy_dropdown.value]
        
        title_parts = [f"{self.aggregation_dropdown.value} - {hierarchy_desc}"]
        
        if self.region_dropdown.value != 'Todas':
            title_parts.append(f"Região: {self.region_dropdown.value}")
        if self.uf_dropdown.value != 'Todas':
            title_parts.append(f"UF: {self.uf_dropdown.value}")
        if self.mun_dropdown.value != 'Todas':
            title_parts.append(f"Município: {self.mun_dropdown.value}")
            
        if self.segment2_dropdown.value != 'Todas':
            title_parts.append(f"por {self.segment2_dropdown.value}")
            if self.segment3_dropdown.value != 'Todas':
                title_parts.append(f"e {self.segment3_dropdown.value}")
        
        return ' - '.join(title_parts)

    def _connect_observers(self):
        """Connect widget observers."""
        try:
            # Connect the submit button to update_plot
            self.submit_button.on_click(self.update_plot)
            
            # Update segmentation options when aggregation changes
            self.aggregation_dropdown.observe(self._update_segmentation_options, names='value')
            
            # Update geographic filters
            self.region_dropdown.observe(self._update_ufs, names='value')
            self.uf_dropdown.observe(self._update_municipios, names='value')
            
            print("Observers connected successfully")
            print("Ready to use - click 'Atualizar Gráfico' to update the visualization")
            
        except Exception as e:
            print(f"Error connecting observers: {str(e)}")
            print(traceback.format_exc())

    def _update_segmentation_options(self, change):
        """Handler for changes in aggregation selection"""
        try:
            new_agg = change.new
            with self.debug_output:
                print(f"\nAgregação alterada para: {new_agg}")
            
            # Update segmentation dropdowns with new options
            current_level = self.hierarchy_dropdown.value
            
            # Only update hierarchy options if current value is not in new options
            if current_level not in self.hierarchy_levels:
                self.hierarchy_dropdown.value = self.hierarchy_levels[0]
            
            self.segment2_dropdown.options = ['Todas']
            self.segment2_dropdown.value = 'Todas'
            
            self.segment3_dropdown.options = ['Todas']
            self.segment3_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in aggregation change handler: {str(e)}")
            self.debug_print(traceback.format_exc())

    def _update_ufs(self, change):
        """Handler para mudanças na região"""
        try:
            #self.debug_print(f"\nRegião alterada para: {change.new}")
                
            if change.new == 'Todas':
                self.uf_dropdown.options = ['Todas']
            else:
                # Debug print the data for this region
                mask = self.df['parent_regiao'].str.upper() == change.new.upper()
                #self.debug_print(f"Number of rows for region {change.new}: {mask.sum()}")
                
                # Get UFs for selected region from dataframe
                ufs = ['Todas'] + sorted([uf for uf in self.df[mask]['parent_uf'].unique() if pd.notna(uf)])
                #self.debug_print(f"UFs found: {ufs}")
                
                self.uf_dropdown.options = ufs
            
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in region change handler: {str(e)}")
            self.debug_print(traceback.format_exc())

    def _update_municipios(self, change):
        """Handler para mudanças na UF"""
        try:
            #self.debug_print(f"\nUF alterada para: {change.new}")
                
            if change.new == 'Todas':
                self.mun_dropdown.options = ['Todas']
            else:
                # Debug print the data for this UF
                mask = (
                    (self.df['parent_regiao'].str.upper() == self.region_dropdown.value.upper()) &
                    (self.df['parent_uf'].str.upper() == change.new.upper())
                )
                #self.debug_print(f"Number of rows for UF {change.new}: {mask.sum()}")
                
                # Get municipalities for selected UF from dataframe
                municipios = ['Todas'] + sorted([m for m in self.df[mask]['parent_municipio'].unique() if pd.notna(m)])
                #self.debug_print(f"Municipalities found: {len(municipios)-1}")
                
                self.mun_dropdown.options = municipios
            
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in UF change handler: {str(e)}")
            self.debug_print(traceback.format_exc())

    def _on_mun_change(self, change):
        """Handler para mudanças no município"""
        try:
            self.debug_print(f"\nMunicípio alterado para: {change.new}")
        except Exception as e:
            self.debug_print(f"Error in municipality change handler: {str(e)}")
            self.debug_print(traceback.format_exc()) 