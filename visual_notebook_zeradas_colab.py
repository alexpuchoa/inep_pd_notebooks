import os
os.environ["PYTHONIOENCODING"] = "cp1252"

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import traceback

# Set the default renderer for Plotly to work in Colab
pio.renderers.default = 'colab'

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
            
            # Create widgets and display interface
            self._create_widgets()
            self.display_interface()
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            print(traceback.format_exc())

    def _create_widgets(self):
        """Create the control widgets."""
        try:
            print("- Creating aggregation dropdown...")
            self.aggregation_dropdown = widgets.Dropdown(
                options=self.aggregation_options,
                value=self.aggregation_options[0],
                description='Agregação:'
            )
            
            print("- Creating hierarchy dropdown...")
            self.hierarchy_dropdown = widgets.Dropdown(
                options=self.hierarchy_levels,
                value=self.hierarchy_levels[0],
                description='Nível:'
            )
            
            print("- Creating segmentation dropdowns...")
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
            self.epsilon_dropdown = widgets.Dropdown(
                options=sorted(self.df['epsilon'].unique()),
                description='Epsilon:'
            )
            
            self.delta_dropdown = widgets.Dropdown(
                options=sorted(self.df['delta'].unique()),
                description='Delta:'
            )
            
            print("- Creating submit button...")
            self.submit_button = widgets.Button(
                description='Atualizar Gráfico',
                button_style='primary',
                tooltip='Clique para atualizar o gráfico'
            )
            
            print("- Creating plot output...")
            self.plots_output = widgets.Output(
                layout=widgets.Layout(
                    height='1000px',  # Height for both plots
                    width='100%',
                    border='1px solid #ddd'
                )
            )
            
            print("All widgets created successfully!")
            
        except Exception as e:
            print(f"Error creating widgets: {str(e)}")
            print(traceback.format_exc())

    def display_interface(self):
        """Display the interface components."""
        try:
            print("\nDisplaying interface components:")
            
            # Create containers for each section
            title_section = widgets.VBox([
                widgets.HTML("<h2>Visualização de Valores Perdidos</h2>"),
                widgets.HTML("<p>NIVEL - Como os dados zerados do nivel hierárquico logo abaixo serão agregados.</p>")
            ])
            
            query_section = widgets.VBox([
                widgets.HTML("<b>Configuração da Query</b>"),
                widgets.HBox([self.aggregation_dropdown, self.hierarchy_dropdown])
            ])
            
            segmentation_section = widgets.VBox([
                widgets.HTML("<b>Segmentações</b>"),
                widgets.HBox([self.segment2_dropdown, self.segment3_dropdown])
            ])
            
            dp_params_section = widgets.VBox([
                widgets.HTML("<b>Parâmetros DP</b>"),
                widgets.HBox([self.epsilon_dropdown, self.delta_dropdown])
            ])
            
            geo_filters_section = widgets.VBox([
                widgets.HTML("<b>Filtros Geográficos</b>"),
                widgets.HBox([self.region_dropdown, self.uf_dropdown, self.mun_dropdown])
            ])
            
            # Create main container
            main_container = widgets.VBox([
                title_section,
                query_section,
                segmentation_section,
                dp_params_section,
                geo_filters_section,
                self.submit_button,
                widgets.HTML("<h3>Gráficos:</h3>"),
                self.plots_output
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

    def update_plot(self, button_clicked=None):
        """Update plot with the current selections."""
        try:
            print("\nUpdating plots...")
            
            # Get current selections
            aggregation = self.aggregation_dropdown.value
            hierarchy = self.hierarchy_dropdown.value
            seg2 = self.segment2_dropdown.value
            seg3 = self.segment3_dropdown.value
            epsilon = self.epsilon_dropdown.value
            delta = self.delta_dropdown.value
            region = self.region_dropdown.value
            uf = self.uf_dropdown.value
            mun = self.mun_dropdown.value
            
            # Debug print
            print(f"Filtering with: agg={aggregation}, eps={epsilon}, delta={delta}")
            print(f"Geographic filters: region={region}, uf={uf}, mun={mun}")
            
            # Filter data based on selections
            filtered_df = self.df[
                (self.df['aggregated_data'].str.upper() == aggregation.upper()) &
                (self.df['epsilon'] == epsilon) &
                (self.df['delta'] == delta)
            ]
            
            # Apply geographic filters if selected
            if region != 'Todas':
                filtered_df = filtered_df[filtered_df['NO_REGIAO'] == region]
            if uf != 'Todas':
                filtered_df = filtered_df[filtered_df['SG_UF'] == uf]
            if mun != 'Todas':
                filtered_df = filtered_df[filtered_df['NO_MUNICIPIO'] == mun]
            
            print(f"Filtered data shape: {filtered_df.shape}")
            
            # Group by epsilon and calculate statistics
            df_plot = filtered_df.groupby('epsilon').agg({
                'original_value': ['count', 'sum'],
                'dp_avg': ['count']
            }).reset_index()
            
            # Calculate percentages and totals
            df_plot['percentage'] = (
                (df_plot[('dp_avg', 'count')] - df_plot[('original_value', 'count')]) / 
                df_plot[('original_value', 'count')] * 100
            )
            df_plot['total_lost'] = (
                df_plot[('original_value', 'count')] - 
                df_plot[('dp_avg', 'count')]
            )
            
            # Create percentage plot
            fig_percentages = go.Figure()
            fig_percentages.add_trace(go.Bar(
                x=df_plot['epsilon'].astype(str),
                y=df_plot['percentage'],
                name='Percentual',
                text=df_plot['percentage'].round(2),
                textposition='auto',
            ))
            fig_percentages.update_layout(
                title='Percentual de Valores Perdidos',
                xaxis_title='Epsilon',
                yaxis_title='Percentual (%)',
                showlegend=True,
                height=400,
                width=900
            )
            
            # Create totals plot
            fig_totals = go.Figure()
            fig_totals.add_trace(go.Bar(
                x=df_plot['epsilon'].astype(str),
                y=df_plot['total_lost'],
                name='Total',
                text=df_plot['total_lost'].round(0),
                textposition='auto',
            ))
            fig_totals.update_layout(
                title='Total de Valores Perdidos',
                xaxis_title='Epsilon',
                yaxis_title='Total',
                showlegend=True,
                height=400,
                width=900
            )
            
            # Display plots
            with self.plots_output:
                clear_output(wait=True)
                fig_percentages.show()
                fig_totals.show()
                print("Plots updated successfully!")
            
        except Exception as e:
            print(f"Error updating plots: {str(e)}")
            print(traceback.format_exc())

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
            print(f"Error in aggregation change handler: {str(e)}")
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

                if self.region_dropdown.value != 'Todas':
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

                if self.uf_dropdown.value != 'Todas':
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

                if self.mun_dropdown.value != 'Todas':
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
                if self.mun_dropdown.value != 'Todas':
                    mask = mask & (self.df['parent_municipio'] == self.mun_dropdown.value)
                    
                filtered_df = self.df[mask]
                group_col = 'parent_municipio'
            
            else:
                print(f"Invalid hierarchy level: {hierarchy_level}")
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
            print(f"Filtered data shape: {filtered_df.shape[0]}")
            print(f"Results shape: {results.shape}")
            
            return results.sort_values('group_by_val1')
            
        except Exception as e:
            print(f"Error in get_lost_values_results: {str(e)}")
            print(traceback.format_exc())
            return pd.DataFrame(columns=['group_by_val1', 'lost_entities', 'total_entities', 
                                      'median_original', 'median_original_lost'])

    def plot_lost_values(self, results):
        """
        Plota dois gráficos separados usando VBox.
        """
        try:
            if results.empty:
                print("No data to plot")
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
            print(f"Error plotting: {str(e)}")
            print(traceback.format_exc())

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