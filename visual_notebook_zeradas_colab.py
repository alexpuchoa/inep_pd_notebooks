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
            if hierarchy == 'NO_REGIAO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'SG_UF']
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]

                # Group by parent_regiao and calculate statistics
                df_plot = filtered_df.groupby('group_by_val1').agg({
                    # Count all rows in group
                    'group_by_val1': 'count',  # Simple count for total entities
                    # Count rows where dp_avg is 0 or NULL or lost > 0
                    'dp_avg': lambda x: ((x == 0.0) | x.isna() | (filtered_df['lost'] > 0)).sum()
                }).reset_index()

            elif hierarchy == 'SG_UF':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'NO_MUNICIPIO']
                if region != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_regiao'] == region]
                if uf != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == uf]

                # Group by parent_uf and calculate statistics
                df_plot = filtered_df.groupby('parent_uf').agg({
                    # Count all rows in group
                    'group_by_val1': 'count',  # Simple count for total entities
                    # Count rows where dp_avg is 0 or NULL or lost > 0
                    'dp_avg': lambda x: ((x == 0.0) | x.isna() | (filtered_df['lost'] > 0)).sum()
                }).reset_index()

            elif hierarchy == 'NO_MUNICIPIO':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                if uf != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_uf'] == uf]
                if mun != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_municipio'] == mun]

                # Group by parent_municipio and calculate statistics
                df_plot = filtered_df.groupby('parent_municipio').agg({
                    # Count all rows in group
                    'group_by_val1': 'count',  # Simple count for total entities
                    # Count rows where dp_avg is 0 or NULL or lost > 0
                    'dp_avg': lambda x: ((x == 0.0) | x.isna() | (filtered_df['lost'] > 0)).sum()
                }).reset_index()

            elif hierarchy == 'CO_ENTIDADE':
                filtered_df = filtered_df[filtered_df['group_by_col1'].str.upper() == 'CO_ENTIDADE']
                if mun != 'Todas':
                    filtered_df = filtered_df[filtered_df['parent_municipio'] == mun]
            
                # Group by parent_municipio and calculate statistics
                df_plot = filtered_df.groupby('parent_municipio').agg({
                    # Count all rows in group
                    'group_by_val1': 'count',  # Simple count for total entities
                    # Count rows where dp_avg is 0 or NULL or lost > 0
                    'dp_avg': lambda x: ((x == 0.0) | x.isna() | (filtered_df['lost'] > 0)).sum()
                }).reset_index()

            # Rename columns consistently
            df_plot.columns = ['group_by_val1', 'total_entities', 'lost_entities']
            
            print(f"Filtered data shape: {filtered_df.shape}")
           
            # Calculate percentages and totals
            df_plot['percentage'] = (
                df_plot['lost_entities'] / 
                df_plot['total_entities'] * 100
            )
            df_plot['total_lost'] = df_plot['lost_entities']
            
            # Create single figure with secondary y-axis
            fig = go.Figure()

            # Add bar plot for totals (primary y-axis)
            fig.add_trace(go.Bar(
                x=df_plot['group_by_val1'],
                y=df_plot['lost_entities'],
                name='Total Perdidos',
                text=df_plot['lost_entities'].round(0),
                textposition='auto',
            ))

            # Add line plot for percentages (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=df_plot['group_by_val1'],
                y=df_plot['percentage'],
                name='Percentual',
                text=df_plot['percentage'].round(2),
                textposition='top center',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='red')
            ))

            # Update layout with secondary y-axis
            fig.update_layout(
                title=f'Valores Perdidos por {hierarchy}',
                xaxis_title=hierarchy,
                yaxis_title='Total de Entidades Perdidas',
                yaxis2=dict(
                    title='Percentual (%)',
                    overlaying='y',
                    side='right',
                    rangemode='tozero'
                ),
                showlegend=True,
                height=600,
                width=1000,
                margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Clear the output and show the plot
            self.plots_output.clear_output(wait=True)
            with self.plots_output:
                fig.show()
            
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