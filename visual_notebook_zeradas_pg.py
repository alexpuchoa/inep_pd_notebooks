import os
os.environ["PYTHONIOENCODING"] = "cp1252"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import logging
from psycopg2.extras import RealDictCursor
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

DB_PARAMS = {
    'dbname': 'inep_synthetic_data',
    'user': 'postgres',
    'password': '0alxndr7',
    'host': 'localhost',
    'port': '5432'
}

TABLE_NAMES = {
    'db_results': 'dp_results_bq',
    'db_no_dp_results': 'no_dp_results_bq',
    'db_results_stats': 'dp_results_stats_bq'
}



class LostValuesVisualization:
    """Visualização específica para contagem de valores perdidos."""
    
    def __init__(self, conn_params):
        """Initialize the visualization interface."""
        try:
            self.conn_params = conn_params
            
            # Load queries configuration
            notebook_dir = Path().absolute()
            queries_file = notebook_dir / "queries_formatadas_bq.csv"
            self.queries_config = pd.read_csv(queries_file, sep=';')
            
            # Define ordered lists - adding back NO_REGIAO
            self.hierarchy_levels = ['NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'CO_ENTIDADE']
            self.ordered_aggregations = ['QT_ALUNOS', 'MEDIA_NOTA', 'SOMA_NOTAS']
            
            # Extract unique aggregated_data values in specified order
            self.aggregation_options = [agg for agg in self.ordered_aggregations 
                                      if agg in self.queries_config['aggregated_data'].str.upper().unique()]
            
            # Create mapping of segmentation options for each aggregation
            self.segmentation_map = {}
            for agg in self.aggregation_options:
                agg_queries = self.queries_config[self.queries_config['aggregated_data'].str.upper() == agg]
                
                # Parse group_by column to get segmentation options
                group_by_values = agg_queries['group_by'].dropna().unique()
                
                # Initialize lists for each segmentation level
                seg2_options = set()
                seg3_options = set()
                
                for group_by in group_by_values:
                    if pd.isna(group_by):
                        continue
                    
                    # Split group_by string into individual columns
                    cols = [col.strip() for col in group_by.split(',')]
                    
                    # Add other segmentations if they exist
                    if len(cols) > 1:
                        seg2_options.add(cols[1])
                    if len(cols) > 2:
                        seg3_options.add(cols[2])
                
                # Store options for this aggregation
                self.segmentation_map[agg] = {
                    'seg1': self.hierarchy_levels,  # Use ordered hierarchy levels
                    'seg2': sorted(seg2_options | {'Todas'}),
                    'seg3': sorted(seg3_options | {'Todas'})
                }
            
            self._create_widgets()
            
            # Initialize geographic filters
            self._load_regions()
            
            self._connect_observers()
            
        except Exception as e:
            print(f"Error initializing visualization: {str(e)}")
            print(traceback.format_exc())

    def _create_widgets(self):
        """
        Cria os widgets com nova estrutura baseada em agregações e segmentações.
        """
        try:
            # Debug output
            self.debug_output = widgets.Output(
                layout=widgets.Layout(
                    height='200px',
                    width='100%',
                    border='2px solid #ccc',
                    padding='10px',
                    margin='10px 0',
                    overflow_y='auto'
                )
            )
            
            # Aggregation type dropdown
            self.aggregation_dropdown = widgets.Dropdown(
                options=self.aggregation_options,
                value=self.aggregation_options[0],
                description='Agregação:'
            )
            
            # Hierarchical level dropdown (first segmentation)
            self.hierarchy_dropdown = widgets.Dropdown(
                options=self.segmentation_map[self.aggregation_options[0]]['seg1'],
                value=self.segmentation_map[self.aggregation_options[0]]['seg1'][0],
                description='Nível:'
            )
            
            # Second segmentation dropdown
            self.segment2_dropdown = widgets.Dropdown(
                options=self.segmentation_map[self.aggregation_options[0]]['seg2'],
                value='Todas',
                description='Segm. 2:'
            )
            
            # Third segmentation dropdown
            self.segment3_dropdown = widgets.Dropdown(
                options=self.segmentation_map[self.aggregation_options[0]]['seg3'],
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
            
            # DP parameters
            self.epsilon_dropdown = widgets.Dropdown(
                options=[1.0, 5.0, 10.0],
                value=1.0,
                description='Epsilon:'
            )
            
            self.delta_dropdown = widgets.Dropdown(
                options=[1e-5, 1e-2],
                value=1e-5,
                description='Delta:'
            )
            
            # Submit button
            self.submit_button = widgets.Button(
                description='Atualizar Gráfico',
                button_style='primary',
                tooltip='Clique para atualizar o gráfico com as seleções atuais'
            )
            
            # Initialize empty figures for the VBox
            self.fig_percentages = go.FigureWidget()
            self.fig_totals = go.FigureWidget()
            
            # Create VBox with both figures
            self.lost_values_fig = widgets.VBox([
                self.fig_percentages,
                self.fig_totals
            ])
            
            # Test debug output
            #with self.debug_output:
            #    print("Widgets created successfully")
            #    print(f"Available aggregations: {self.aggregation_options}")
                
        except Exception as e:
            print(f"Error creating widgets: {str(e)}")
            print(traceback.format_exc())

    def display_chart(self):
        """
        Display the visualization interface with improved layout
        """
        try:
            # Title
            
            title = widgets.HTML(
                
                "<h2 style='text-align: center; margin: 20px 0; color: #2c3e50; font-family: Arial, sans-serif; padding: 15px; border-bottom: 2px solid #3498db;'>Visualização de Valores Perdidos</h2>"
            )

            notas = widgets.HTML(
                "<h4 style='text-align: left; margin: 15px 0; color: #2c3e50; font-family: Arial, sans-serif;'>NIVEL - Como os dados zerados do nivel hirárquico logo abaixo serão agregados. Ex.: Nivel = SG_UF, dados de MUNICIPIOS zerados são agregados por UF.<br>SEGMENTAÇÕES 2 e 3 - Somente disponível para NIVEL = NO_REGIAO.<br>FILTRO GEOGRÁFICO - Seleção obrigatória se NIVEL <> NO_REGIAO. Se NIVEL = SG_UF, uma Região deve ser selecionada. Se NIVEL = CO_ENTIDADE, um Município deve ser selecionado.</br></h4>"
            )
            
            # Controls in sections
            controls_title = widgets.HTML("<h3 style='color: #2c3e50; font-family: Arial, sans-serif; margin: 15px 0;'>Controles</h3>")
            
            query_controls = widgets.VBox([
                widgets.HTML("<b>Configuração da Query</b>"),
                widgets.HBox([
                    self.aggregation_dropdown,
                    self.hierarchy_dropdown
                ])
            ])
            
            segmentation_controls = widgets.VBox([
                widgets.HTML("<b>Segmentações</b>"),
                widgets.HBox([
                    self.segment2_dropdown,
                    self.segment3_dropdown
                ])
            ])
            
            param_controls = widgets.VBox([
                widgets.HTML("<b>Parâmetros DP</b>"),
                widgets.HBox([
                    self.epsilon_dropdown,
                    self.delta_dropdown
                ])
            ])
            
            geo_controls = widgets.VBox([
                widgets.HTML("<b>Filtros Geográficos</b>"),
                widgets.HBox([
                    self.region_dropdown,
                    self.uf_dropdown,
                    self.mun_dropdown
                ])
            ])
            
            # Main container
            self.container = widgets.VBox([
                title,
                notas,
                controls_title,
                query_controls,
                segmentation_controls,
                param_controls,
                geo_controls,
                self.submit_button,
                self.lost_values_fig,
                widgets.HTML("<h4>Debug Messages:</h4>"),
                self.debug_output
            ], layout=widgets.Layout(
                padding='20px',
                width='100%',
                border='1px solid #ddd',
                margin='10px'
            ))
            
            # Display the interface
            display(self.container)
            
        except Exception as e:
            print(f"Error displaying interface: {str(e)}")
            print(traceback.format_exc())

    def _load_regions(self):
        """Load initial list of regions from reference CSV"""
        try:
            # Load reference data
            notebook_dir = Path().absolute()
            ref_file = notebook_dir / "regiao_uf_municipio_escola.csv"
            ref_data = pd.read_csv(ref_file, sep=';')
            
            # Get unique regions
            regions = ['Todas'] + sorted(ref_data['NO_REGIAO'].unique().tolist())
            self.region_dropdown.options = regions
            
            self.debug_print("Regions loaded successfully from reference file")
            
        except Exception as e:
            self.debug_print(f"Error loading regions: {str(e)}")
            self.debug_print(traceback.format_exc())

    def update_plot(self, button_clicked=None):
        """
        Atualiza o gráfico de valores perdidos.
        """
        try:
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

            self.debug_print("\n=== Starting Update ===")
            self.debug_print(f"Aggregation: {self.aggregation_dropdown.value}")
            self.debug_print(f"Hierarchy: {hierarchy_level}")
            self.debug_print(f"Epsilon: {self.epsilon_dropdown.value}")
            self.debug_print(f"Delta: {self.delta_dropdown.value}")
            self.debug_print(f"Region: {self.region_dropdown.value}")
            self.debug_print(f"UF: {self.uf_dropdown.value}")
            self.debug_print(f"Municipality: {self.mun_dropdown.value}")
            
            # Get results only if validations pass
            results = self.get_lost_values_results()
            
            if results is None or results.empty:
                self.debug_print("No results to plot")
                return
            
            self.debug_print(f"Retrieved {len(results)} rows")
            self.plot_lost_values(results)
            
        except Exception as e:
            self.debug_print(f"Error updating plot: {str(e)}")
            self.debug_print(traceback.format_exc())

    def debug_print(self, *messages):
        """Helper method to print debug messages. Can handle multiple arguments."""
        with self.debug_output:
            print(*messages)

    def get_lost_values_results(self):
        """
        Obtém os resultados do banco de dados incluindo contagem de valores perdidos e medianas.
        """
        try:
            hierarchy_level = self.hierarchy_dropdown.value
            
            # Debug print current selections
            #self.debug_print("\nCurrent selections:")
            #self.debug_print(f"Aggregation: {self.aggregation_dropdown.value}")
            #self.debug_print(f"Hierarchy: {hierarchy_level}")
            #self.debug_print(f"Segmentation 2: {self.segment2_dropdown.value}")
            #self.debug_print(f"Segmentation 3: {self.segment3_dropdown.value}")
            #self.debug_print(f"Region: {self.region_dropdown.value}")
            
            # Build base conditions list
            base_conditions = [
                "upper(aggregated_data) = %s",
                "epsilon = %s",
                "delta = %s"
            ]
            
            # Build base parameters list
            base_params = [
                self.aggregation_dropdown.value.upper(),
                self.epsilon_dropdown.value,
                self.delta_dropdown.value
            ]
          
            # Build appropriate query based on hierarchy level
            if hierarchy_level == 'NO_REGIAO':
                print('REGIÃO level. Searching for sg_uf')
                base_conditions.extend([
                    "upper(group_by_col1) = 'SG_UF'"
                ])
                
                # Add segmentation conditions only if they are not 'Todas'
                if self.segment2_dropdown.value != 'Todas':
                    base_conditions.append("group_by_col2 = %s")
                    base_params.append(self.segment2_dropdown.value)
                    
                    # Only consider seg3 if seg2 is selected
                    if self.segment3_dropdown.value != 'Todas':
                        base_conditions.append("group_by_col3 = %s")
                        base_params.append(self.segment3_dropdown.value)
                    else:
                        base_conditions.append("group_by_col3 = ''")
                else:
                    base_conditions.append("group_by_col2 = ''")
                    base_conditions.append("group_by_col3 = ''")

                if self.region_dropdown.value != 'Todas':
                    # A specific region has been selected
                    base_conditions.append("parent_regiao = %s")
                    base_params.append(self.region_dropdown.value)
                    
                # Debug print conditions and parameters
                #self.debug_print("\nQuery conditions:")
                #self.debug_print(base_conditions)
                #self.debug_print("\nQuery parameters:")
                #self.debug_print(base_params)
                
                query = f"""
                    SELECT 
                        parent_regiao as group_by_val1,
                        COUNT(*) as total_entities,
                        COUNT(*) FILTER (
                            WHERE dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0
                        ) as lost_entities,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY original_value) as median_original,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE 
                            WHEN dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0 THEN original_value 
                        END) as median_original_lost
                    FROM dp_results_stats_bq
                    WHERE {' AND '.join(base_conditions)}
                    GROUP BY parent_regiao
                    ORDER BY parent_regiao
                """
                
            elif hierarchy_level == 'SG_UF':
                print('UF level. Searching for NO_MUNICIPIO')
                base_conditions.extend([
                    "upper(parent_regiao) = %s",
                    "upper(group_by_col1) = 'NO_MUNICIPIO'"
                ])
                base_params.append(self.region_dropdown.value.upper())

                if self.uf_dropdown.value != 'Todas':
                    # A specific UF has been selected
                    base_conditions.append("parent_uf = %s")
                    base_params.append(self.uf_dropdown.value)
                
                # Debug print conditions and parameters
                #self.debug_print("\nQuery conditions:")
                #self.debug_print(base_conditions)
                #self.debug_print("\nQuery parameters:")
                #self.debug_print(base_params)
                
                query = f"""
                    SELECT 
                        parent_uf as group_by_val1,
                        COUNT(*) as total_entities,
                        COUNT(*) FILTER (
                            WHERE dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0
                        ) as lost_entities,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY original_value) as median_original,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE 
                            WHEN dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0 THEN original_value 
                        END) as median_original_lost
                    FROM dp_results_stats_bq
                    WHERE {' AND '.join(base_conditions)}
                    GROUP BY parent_uf
                    ORDER BY parent_uf
                """
                
            elif hierarchy_level == 'NO_MUNICIPIO':
                print('Municipio level. Searching for CO_ENTIDADE')
                base_conditions.extend([
                    "upper(parent_regiao) = %s",
                    "upper(parent_uf) = %s",
                    "upper(group_by_col1) = 'CO_ENTIDADE'"
                ])
                base_params.extend([
                    self.region_dropdown.value.upper(),
                    self.uf_dropdown.value.upper()
                ])
                
                # Add segmentation conditions only if they are not 'Todas'
                if self.segment2_dropdown.value != 'Todas':
                    base_conditions.append("group_by_col2 = %s")
                    base_params.append(self.segment2_dropdown.value)
                    
                    # Only consider seg3 if seg2 is selected
                    if self.segment3_dropdown.value != 'Todas':
                        base_conditions.append("group_by_col3 = %s")
                        base_params.append(self.segment3_dropdown.value)
                    else:
                        base_conditions.append("group_by_col3 = ''")
                else:
                    base_conditions.append("group_by_col2 = ''")
                    base_conditions.append("group_by_col3 = ''")

                if self.mun_dropdown.value != 'Todas':
                    # A specific municipality has been selected
                    base_conditions.append("parent_municipio = %s")
                    base_params.append(self.mun_dropdown.value)
                
                query = f"""
                    SELECT 
                        parent_municipio as group_by_val1,
                        COUNT(*) as total_entities,
                        COUNT(*) FILTER (WHERE dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0) as lost_entities,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY original_value) as median_original,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CASE 
                            WHEN dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0 THEN original_value 
                        END) as median_original_lost
                    FROM dp_results_stats_bq
                    WHERE {' AND '.join(base_conditions)}
                    GROUP BY parent_municipio
                    ORDER BY parent_municipio
                """
                
            elif hierarchy_level == 'CO_ENTIDADE':
                print('Escola level. Searching for CO_ENTIDADE')
                base_conditions.extend([
                    "upper(parent_regiao) = %s",
                    "upper(parent_uf) = %s",
                    "upper(parent_municipio) = %s",
                    "upper(group_by_col1) = 'CO_ENTIDADE'"
                ])
                base_params.extend([
                    self.region_dropdown.value.upper(),
                    self.uf_dropdown.value.upper(),
                    self.mun_dropdown.value.upper()
                ])

                # Add segmentation conditions only if they are not 'Todas'
                if self.segment2_dropdown.value != 'Todas':
                    base_conditions.append("group_by_col2 = %s")
                    base_params.append(self.segment2_dropdown.value)
                    
                    # Only consider seg3 if seg2 is selected
                    if self.segment3_dropdown.value != 'Todas':
                        base_conditions.append("group_by_col3 = %s")
                        base_params.append(self.segment3_dropdown.value)
                    else:
                        base_conditions.append("group_by_col3 = ''")
                else:
                    base_conditions.append("group_by_col2 = ''")
                    base_conditions.append("group_by_col3 = ''")

                if self.mun_dropdown.value != 'Todas':
                    # A specific municipality has been selected
                    base_conditions.append("parent_municipio = %s")
                    base_params.append(self.mun_dropdown.value)

                query = f"""
                    SELECT 
                        group_by_val1,
                        1 as total_entities,
                        CASE WHEN dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0 THEN 1 ELSE 0 END as lost_entities,
                        original_value as median_original,
                        CASE WHEN dp_avg = 0.0 OR dp_avg IS NULL OR lost > 0 THEN original_value ELSE NULL END as median_original_lost
                    FROM dp_results_stats_bq
                    WHERE {' AND '.join(base_conditions)}
                    ORDER BY group_by_val1
                """

            # Debug print the actual SQL query with parameters
            debug_query = query
            for param in base_params:
                debug_query = debug_query.replace('%s', repr(param), 1)
            print("\nActual SQL Query to be executed:")
            print(debug_query)
            print("\n")
            
            # Execute query
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    self.debug_print("\nExecuting query with parameters:", base_params)
                    cur.execute(query, base_params)
                    
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    df = pd.DataFrame(rows, columns=columns)
                    
                    #self.debug_print(f"\nRetrieved {len(df)} rows")
                    #self.debug_print("\nResults preview:")
                    #self.debug_print(df.head().to_string())
                    
                    return df
                    
        except Exception as e:
            self.debug_print(f"Database error: {str(e)}")
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
            self.fig_percentages.data = []
            self.fig_totals.data = []
            
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
                    marker_color='rgb(158,202,225)',
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
        """
        Conecta os observadores aos widgets.
        """
        try:
            # Connect aggregation dropdown to update segmentation options
            self.aggregation_dropdown.observe(self._on_aggregation_change, names='value')
            
            # Connect geographic filters
            self.region_dropdown.observe(self._on_region_change, names='value')
            self.uf_dropdown.observe(self._on_uf_change, names='value')
            self.mun_dropdown.observe(self._on_mun_change, names='value')
            
            # Connect submit button
            self.submit_button.on_click(self.update_plot)
            
            with self.debug_output:
                print("Observers connected successfully")
                print("Ready to use - click 'Atualizar Gráfico' to update the visualization")
            
        except Exception as e:
            print(f"Error connecting observers: {str(e)}")
            print(traceback.format_exc())

    def _on_aggregation_change(self, change):
        """Handler for changes in aggregation selection"""
        try:
            new_agg = change.new
            with self.debug_output:
                print(f"\nAgregação alterada para: {new_agg}")
                print(f"Opções de segmentação disponíveis: {self.segmentation_map[new_agg]}")
            
            # Update segmentation dropdowns with new options
            current_level = self.hierarchy_dropdown.value
            
            # Only update hierarchy options if current value is not in new options
            if current_level not in self.hierarchy_levels:
                self.hierarchy_dropdown.value = self.hierarchy_levels[0]
            
            self.segment2_dropdown.options = self.segmentation_map[new_agg]['seg2']
            self.segment2_dropdown.value = 'Todas'
            
            self.segment3_dropdown.options = self.segmentation_map[new_agg]['seg3']
            self.segment3_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in aggregation change handler: {str(e)}")
            self.debug_print(traceback.format_exc())

    def _on_region_change(self, change):
        """Handler para mudanças na região"""
        try:
            with self.debug_output:
                print(f"\nRegião alterada para: {change.new}")
                
            if change.new == 'Todas':
                self.uf_dropdown.options = ['Todas']
            else:
                # Get UFs for selected region from database
                with psycopg2.connect(**self.conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT DISTINCT parent_uf 
                            FROM dp_results_stats_bq 
                            WHERE parent_regiao = %s 
                            ORDER BY parent_uf
                        """, (change.new.upper(),))
                        ufs = ['Todas'] + [row[0] for row in cur.fetchall()]
                        self.uf_dropdown.options = ufs
            
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in region change handler: {str(e)}")

    def _on_uf_change(self, change):
        """Handler para mudanças na UF"""
        try:
            with self.debug_output:
                print(f"\nUF alterada para: {change.new}")
                
            if change.new == 'Todas':
                self.mun_dropdown.options = ['Todas']
            else:
                # Get municipalities for selected UF from database
                with psycopg2.connect(**self.conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT DISTINCT parent_municipio 
                            FROM dp_results_stats_bq 
                            WHERE parent_uf = %s 
                            ORDER BY parent_municipio
                        """, (change.new.upper(),))
                        municipalities = ['Todas'] + [row[0] for row in cur.fetchall()]
                        self.mun_dropdown.options = municipalities
            
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error in UF change handler: {str(e)}")

    def _on_mun_change(self, change):
        """Handler para mudanças no município"""
        try:
            with self.debug_output:
                print(f"\nMunicípio alterado para: {change.new}")
        except Exception as e:
            self.debug_print(f"Error in municipality change handler: {str(e)}") 