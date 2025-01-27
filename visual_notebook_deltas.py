"""
Módulo para visualização interativa dos resultados com e sem DP em Jupyter notebooks.
"""

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
import plotly.express as px
from pathlib import Path

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
    'db_results': 'dp_results_synthetic',
    'db_no_dp_results': 'no_dp_results',
    'db_results_stats': 'dp_results_stats_synthetic'
}

class VisualizationDeltaEpsilon:
    """
    Classe principal para visualização de dados com Differential Privacy.
    Gerencia a interface interativa e a exibição de gráficos comparativos.
    """

    def __init__(self, conn_params):
        """
        Inicializa a visualização.
        """
        self.conn_params = conn_params
        
        # Initialize current query attributes
        self.current_query_type = 1
        self.current_query_model = 1
        
        # Initialize table names
        self.TABLE_NAMES = {
            'db_results': 'dp_results_synthetic',
            'db_no_dp_results': 'no_dp_results',
            'db_results_stats': 'dp_results_stats_synthetic'
        }
        
        # Create debug output
        self.debug_output = widgets.Output()
        
        # Initialize region data first
        self.region_data = None
        self.uf_by_region = {}
        self.mun_by_uf = {}
        self._load_region_data()
        
        # Load metadata after region data
        self._load_query_metadata()
        
        # Create widgets after data is loaded
        self._create_widgets()
        
        # Connect observers after all widgets are created
        self._connect_observers()
        
        # Initial plot
        self._initial_plot()
        
        self.debug_print("Initializing VisualizationDeltasEpsilons")
        
        # Define the initial aggregation model for the query
        # Used to set the y-axis title for the bars plot
        self.query_model_aggregation = {1: 'Soma Alunos', 2: 'Média Nota', 3: 'Soma Nota'}
        self.selected_model = self.query_model_aggregation[1]

    def debug_print(self, message):
        """
        Imprime mensagens de debug no widget de output.
        """
        with self.debug_output:
            print(message)

    def _connect_observers(self):
        """
        Conecta os observadores aos widgets para reagir às mudanças.
        """
        try:
            # Connect query type and model changes
            self.query_type_slider.observe(self._on_query_type_change, names='value')
            self.query_model_dropdown.observe(self._on_query_model_change, names='value')
            
            # Connect geographic filters for cascading updates
            self.region_dropdown.observe(self._on_region_change, names='value')
            self.uf_dropdown.observe(self._on_uf_change, names='value')
            self.mun_dropdown.observe(self._on_mun_change, names='value')
            
        except Exception as e:
            self.debug_print(f"Error connecting observers: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _on_query_type_change(self, change):
        """
        Handler for query type changes
        """
        self.current_query_type = change.new

    def _on_query_model_change(self, change):
        """
        Handler for query model changes
        """
        self.current_query_model = change.new

    def _on_region_change(self, change):
        """Handler para mudanças na região"""
        try:
            self.debug_print(f"\nRegion changed to {change.new}")
            if change.new == 'Todas':
                # Use pre-computed unique UFs from region_data
                ufs = sorted(self.region_data['SG_UF'].unique())
            else:
                # Use pre-computed dictionary
                ufs = self.uf_by_region.get(change.new, [])
            
            # Update UF dropdown
            self.uf_dropdown.options = ['Todas'] + ufs
            self.uf_dropdown.value = 'Todas'
            
            # Update municipality dropdown based on region
            if change.new == 'Todas':
                muns = sorted(self.region_data['NO_MUNICIPIO'].unique())
            else:
                muns = sorted(set(
                    mun for uf in ufs 
                    for mun in self.mun_by_uf.get(uf, [])
                ))
            
            self.mun_dropdown.options = ['Todas'] + muns
            self.mun_dropdown.value = 'Todas'
            
            self.update_plot(None)  # Force update after region change
            
        except Exception as e:
            self.debug_print(f"Error in _on_region_change: {str(e)}")

    def _on_uf_change(self, change):
        """Handler para mudanças na UF"""
        try:
            self.debug_print(f"\nUF changed to {change.new}")
            if change.new == 'Todas':
                region = self.region_dropdown.value
                if region != 'Todas':
                    # Use pre-computed dictionaries
                    ufs = self.uf_by_region.get(region, [])
                    muns = sorted(set(
                        mun for uf in ufs 
                        for mun in self.mun_by_uf.get(uf, [])
                    ))
                else:
                    muns = sorted(self.region_data['NO_MUNICIPIO'].unique())
            else:
                # Use pre-computed dictionary
                muns = self.mun_by_uf.get(change.new, [])
            
            self.mun_dropdown.options = ['Todas'] + muns
            self.mun_dropdown.value = 'Todas'
            
            self.update_plot(None)  # Force update after UF change
            
        except Exception as e:
            self.debug_print(f"Error in _on_uf_change: {str(e)}")

    def _on_mun_change(self, change):
        """Handler para mudanças no município"""
        try:
            self.debug_print(f"\nMunicipality changed to {change.new}")
            self.update_plot(None)  # Force update after municipality change
        except Exception as e:
            self.debug_print(f"Error in _on_mun_change: {str(e)}")

    def _on_parameter_change(self, change):
        """Handler para mudanças nos parâmetros epsilon/delta/stats"""
        #self.debug_print(f"\nDEBUG: Parameter {change.owner.description} changed to {change.new}")
        self.update_plot(None)

    def display_stats_chart(self):
        """
        Display the visualization interface with improved layout
        """
        # Create three rows of controls
        top_controls = widgets.HBox([
            self.query_type_slider,
            self.query_model_dropdown,
            self.stat_type_dropdown
        ])
        
        middle_controls = widgets.HBox([
            self.region_dropdown,
            self.uf_dropdown,
            self.mun_dropdown
        ])
        
        privacy_controls = widgets.HBox([
            self.epsilon_dropdown,
            self.delta_dropdown
        ])
        
        # Create container for all elements
        self.stats_container = widgets.VBox([
            top_controls,
            middle_controls,
            privacy_controls,
            self.submit_button,
            self.fig_widget,
            self.debug_output
        ])
        
        # Connect submit button to update function
        self.submit_button.on_click(self._on_submit_click)
        
        # Update chart layout
        chart_layout = dict(
            width=1000,
            height=700,
            margin=dict(t=100, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1
            ),
            template='plotly_white',
            hovermode='x unified'
        )
        
        self.fig_widget.update_layout(**chart_layout)
        
        display(self.stats_container)

    def _on_submit_click(self, b):
        """
        Handler for submit button click
        """
        try:
            # Get current metadata first
            current_metadata = self.query_metadata[
                (self.query_metadata['query_type'] == self.current_query_type) & 
                (self.query_metadata['query_model'] == self.current_query_model)
            ].iloc[0]
            
            group_by = str(current_metadata['group_by']).upper()
            self.debug_print(f"Query group by: {group_by}")
            
            # If this is an entity query, no need to check geographic filters
            if 'ENTIDADE' in group_by:
                self.debug_print("Entity-based query detected, proceeding with representative samples")
                self.update_plot(None)
                return
            
            # For non-entity queries, check geographic filters
            if 'MUNICIPIO' in group_by:
                if self.region_dropdown.value == 'Todas' or self.uf_dropdown.value == 'Todas':
                    self.debug_print("Por favor, selecione uma Região e UF para consultas por município")
                    return
            elif 'UF' in group_by:
                if self.region_dropdown.value == 'Todas':
                    self.debug_print("Por favor, selecione uma Região para consultas por UF")
                    return
            
            # Update the plot
            self.update_plot(None)
            
        except Exception as e:
            self.debug_print(f"Error in submit handler: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def connect_db(self):
        """Conecta ao banco de dados."""
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def get_query(self, query_type, query_model):
        """
        Gera a query SQL baseada no tipo e modelo de consulta.
        """
        #print(f"\nDEBUG: Building query for type={query_type}, model={query_model}")
        
        query = f"""
            SELECT 
                group_by_col1,
                group_by_col2,
                group_by_col3,
                group_by_val1,
                group_by_val2,
                group_by_val3,
                dp_avg,
                dp_median,
                dp_stddev,
                dp_ci_upper - dp_ci_lower as ci,
                mae,
                mape,
                num_runs,
                original_value
            FROM {TABLE_NAMES['db_results_stats']}
            WHERE query_model = %s
            AND query_type = %s
            AND epsilon = %s
            AND delta = %s
            ORDER BY group_by_val1, group_by_val2, group_by_val3
        """
        
        #print("DEBUG: Generated query:\n", query)
        return query

    def update_plot(self, _):
        """
        Atualiza o gráfico com base nos filtros selecionados.
        """
        try:
            # Get current values
            self.current_query_type = self.query_type_slider.value
            self.current_query_model = self.query_model_dropdown.value
            
            # Get results
            results = self.get_results()
            
            if results is not None and not results.empty:
                # Update privacy parameters
                self.update_privacy_params(results)
                
                # Filter results based on selected privacy parameters
                epsilon = self.epsilon_dropdown.value
                delta = self.delta_dropdown.value
                
                if epsilon != 'Todos':
                    results = results[results['epsilon'] == float(epsilon)]
                if delta != 'Todos':
                    results = results[results['delta'] == float(delta)]
                
                # Filter results based on geographic selections
                results = self.filter_results(results)
                
                # Update plots
                if not results.empty:
                    self.update_combined_plot(results)
                    self.update_stats_plot(results, self.stats_dropdown.value)
                else:
                    self.debug_print("No results after filtering")
            else:
                self.debug_print("No results available")
                
        except Exception as e:
            self.debug_print(f"Error updating plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def filter_results(self, results):
        """
        Filtra os resultados respeitando a hierarquia geográfica.
        """
        if results.empty:
            return results

        region = self.region_dropdown.value
        uf = self.uf_dropdown.value
        mun = self.mun_dropdown.value
        
        try:
            filtered_results = results.copy()
            
            self.debug_print(f"\nFiltering results:")
            self.debug_print(f"Selected region: {region}")
            self.debug_print(f"Selected UF: {uf}")
            self.debug_print(f"Selected municipality: {mun}")
            
            # Get current metadata
            current_metadata = self.query_metadata[
                (self.query_metadata['query_type'] == self.current_query_type) & 
                (self.query_metadata['query_model'] == self.current_query_model)
            ].iloc[0]
            
            # Get the group by column from metadata
            group_by = str(current_metadata['group_by']).upper()
            self.debug_print(f"Query group by: {group_by}")
            
            # Filter based on selected geographic level and query's group_by
            if 'MUNICIPIO' in group_by:
                if mun != 'Todas':
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper() == mun.upper()
                    ]
                elif uf != 'Todas':
                    muns_in_uf = set(self.region_data[
                        self.region_data['SG_UF'].str.upper() == uf.upper()
                    ]['NO_MUNICIPIO'].str.upper())
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper().isin(muns_in_uf)
                    ]
                elif region != 'Todas':
                    muns_in_region = set(self.region_data[
                        self.region_data['NO_REGIAO'].str.upper() == region.upper()
                    ]['NO_MUNICIPIO'].str.upper())
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper().isin(muns_in_region)
                    ]
            elif 'UF' in group_by or 'ESTADO' in group_by:
                if uf != 'Todas':
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper() == uf.upper()
                    ]
                elif region != 'Todas':
                    ufs_in_region = set(self.region_data[
                        self.region_data['NO_REGIAO'].str.upper() == region.upper()
                    ]['SG_UF'].str.upper())
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper().isin(ufs_in_region)
                    ]
            elif 'REGIAO' in group_by:
                if region != 'Todas':
                    filtered_results = filtered_results[
                        filtered_results['group_by_val1'].str.upper() == region.upper()
                    ]
            
            self.debug_print(f"Filtered results shape: {filtered_results.shape}")
            self.debug_print(f"Available columns: {filtered_results.columns.tolist()}")
            
            return filtered_results
            
        except Exception as e:
            self.debug_print(f"Error in filtering: {str(e)}")
            return results

    def update_combined_plot(self, results):
        """
        Atualiza o gráfico unificado com linhas para valores originais e estatística selecionada.
        """
        try:
            with self.fig_widget.batch_update():
                self.fig_widget.data = []
                
                if results is None or results.empty:
                    self.debug_print("No results to plot")
                    return
                
                # Update privacy parameter options
                self.update_privacy_params(results)
                
                # Filter results based on selected epsilon and delta
                if self.epsilon_dropdown.value != 'Todos':
                    results = results[results['epsilon'] == float(self.epsilon_dropdown.value)]
                if self.delta_dropdown.value != 'Todos':
                    results = results[results['delta'] == float(self.delta_dropdown.value)]
                
                # Create labels for x-axis and group key
                group_keys = []
                x_labels = []
                
                # Verify required columns exist
                required_columns = ['group_by_col1', 'group_by_val1', 'original_value']
                missing_columns = [col for col in required_columns if col not in results.columns]
                if missing_columns:
                    self.debug_print(f"Missing required columns: {missing_columns}")
                    return
                
                self.debug_print(f"Available columns: {results.columns.tolist()}")
                
                for _, row in results.iterrows():
                    key_parts = []
                    label_parts = []
                    for i in range(1, 4):
                        col = f'group_by_col{i}'
                        val = f'group_by_val{i}'
                        if col in row and pd.notna(row[col]) and str(row[col]) not in ['None', '']:
                            key_parts.append(str(row[val]))
                            label_parts.append(f"{row[val]}")
                    group_key = '|'.join(key_parts)
                    group_keys.append(group_key)
                    x_labels.append(' | '.join(label_parts) if label_parts else 'N/A')
                
                # Add group key to results
                results['group_key'] = group_keys
                
                # Get unique groups and sort by original value
                unique_groups = (results.groupby('group_key')
                               .agg({'original_value': 'first'})
                               .sort_values('original_value', ascending=True))
                
                unique_keys = unique_groups.index.tolist()
                original_values = unique_groups['original_value'].tolist()
                
                # Get labels in the same order as sorted values
                unique_labels = []
                for key in unique_keys:
                    idx = group_keys.index(key)
                    unique_labels.append(x_labels[idx])
                
                # Add original value line
                self.fig_widget.add_trace(
                    go.Scatter(
                        name='Valor Original',
                        x=unique_labels,
                        y=original_values,
                        mode='lines+markers',
                        line=dict(color='black', width=2),
                        marker=dict(size=8)
                    )
                )
                
                # Available statistics and their labels
                stats = {
                    'dp_avg': 'Média',
                    'dp_median': 'Mediana',
                    'dp_stddev': 'Desvio Padrão',
                    'dp_var': 'Variância',
                    'dp_min': 'Mínimo',
                    'dp_max': 'Máximo',
                    'dp_sum': 'Soma',
                    'dp_count': 'Contagem'
                }
                
                # Get selected statistic
                stat_type = self.stat_type_dropdown.value
                if stat_type not in results.columns:
                    self.debug_print(f"Selected statistic {stat_type} not found in results")
                    return
                
                stat_name = stats.get(stat_type, stat_type)
                
                # Colors for different parameter combinations
                colors = px.colors.qualitative.Set1
                
                # Add lines for each epsilon-delta combination
                for i, ((eps, delta), param_results) in enumerate(results.groupby(['epsilon', 'delta'])):
                    # Get values for this parameter combination in the same order
                    param_values = []
                    for key in unique_keys:
                        group_results = param_results[param_results['group_key'] == key]
                        if not group_results.empty:
                            val = group_results[stat_type].iloc[0]
                            param_values.append(val)
                        else:
                            param_values.append(None)
                    
                    self.fig_widget.add_trace(
                        go.Scatter(
                            name=f'{stat_name} (ε={eps}, δ={delta})',
                            x=unique_labels,
                            y=param_values,
                            mode='lines+markers',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8)
                        )
                    )
                
                # Update layout
                self.fig_widget.update_layout(
                    title=dict(
                        text=f'Resultados para Query Type {self.current_query_type}',
                        y=0.95,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    xaxis_title="Grupos",
                    yaxis_title="Valores",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1
                    ),
                    xaxis_tickangle=-45,
                    hovermode='x unified'
                )
                
        except Exception as e:
            self.debug_print(f"Error updating combined plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _create_widgets(self):
        """
        Cria todos os widgets da interface.
        """
        # Query type slider
        self.query_type_slider = widgets.IntSlider(
            value=self.current_query_type,
            min=1,
            max=8,
            step=1,
            description='Query Type:',
            continuous_update=False
        )
        
        # Query model dropdown
        self.query_model_dropdown = widgets.Dropdown(
            options=[1, 2, 3],
            value=self.current_query_model,
            description='Query Model:'
        )
        
        # Region dropdown - populate with unique regions from region_data
        unique_regions = sorted(self.region_data['NO_REGIAO'].unique())
        self.region_dropdown = widgets.Dropdown(
            options=['Todas'] + unique_regions,
            value='Todas',
            description='Região:'
        )
        
        # UF dropdown - start with all UFs
        unique_ufs = sorted(self.region_data['SG_UF'].unique())
        self.uf_dropdown = widgets.Dropdown(
            options=['Todas'] + unique_ufs,
            value='Todas',
            description='UF:'
        )
        
        # Municipality dropdown - start with all municipalities
        unique_muns = sorted(self.region_data['NO_MUNICIPIO'].unique())
        self.mun_dropdown = widgets.Dropdown(
            options=['Todas'] + unique_muns,
            value='Todas',
            description='Município:'
        )
        
        # Epsilon dropdown
        self.epsilon_dropdown = widgets.Dropdown(
            options=['Todos'],
            value='Todos',
            description='Epsilon:'
        )
        
        # Delta dropdown
        self.delta_dropdown = widgets.Dropdown(
            options=['Todos'],
            value='Todos',
            description='Delta:'
        )
        
        # Statistic type dropdown
        self.stat_type_dropdown = widgets.Dropdown(
            options=[
                ('Média', 'dp_avg'),
                ('Mediana', 'dp_median'),
                ('Desvio Padrão', 'dp_stddev'),
                ('Variância', 'dp_var'),
                ('Mínimo', 'dp_min'),
                ('Máximo', 'dp_max'),
                ('Soma', 'dp_sum'),
                ('Contagem', 'dp_count')
            ],
            value='dp_avg',
            description='Estatística:'
        )
        
        # Stats dropdown
        self.stats_dropdown = widgets.Dropdown(
            options=[
                ('MAE', 'mae'),
                ('MAPE', 'mape'),
                ('Média DP', 'dp_avg'),
                ('Mediana DP', 'dp_median'),
                ('Desvio Padrão DP', 'dp_stddev'),
                ('Valor Original', 'original'),
                ('Intervalo de Confiança', 'ci')
            ],
            value='dp_avg',  # Changed initial value from 'mae' to 'dp_avg'
            description='Estatística:'
        )
        
        # Submit button
        self.submit_button = widgets.Button(
            description='Atualizar Gráfico',
            button_style='primary',
            tooltip='Clique para atualizar o gráfico com os filtros selecionados'
        )
        
        # Create empty figure with adjusted height and padding
        self.fig_widget = go.FigureWidget(
            layout=go.Layout(
                height=700,
                width=1000,
                margin=dict(b=50, t=50)
            )
        )

    def _load_region_data(self):
        """
        Carrega os dados de regiões do arquivo CSV.
        """
        try:
            self.debug_print("Loading region data...")
            
            # Get current notebook directory
            notebook_dir = Path().absolute()
            csv_file_path = notebook_dir / "regiao_uf_municipio_escola.csv"
            
            # Read the CSV file
            self.region_data = pd.read_csv(
                csv_file_path,
                encoding='utf-8',
                sep=';'
            )
            
            if self.region_data.empty:
                raise ValueError("No region data found in CSV file")
            
            self.debug_print(f"Loaded {len(self.region_data)} region records")
            
            # Prepare relationship dictionaries
            self.uf_by_region = {}
            self.mun_by_uf = {}
            
            for region in self.region_data['NO_REGIAO'].unique():
                region_data = self.region_data[self.region_data['NO_REGIAO'] == region]
                self.uf_by_region[region] = sorted(region_data['SG_UF'].unique())
                
                for uf in self.uf_by_region[region]:
                    uf_data = region_data[region_data['SG_UF'] == uf]
                    self.mun_by_uf[uf] = sorted(uf_data['NO_MUNICIPIO'].unique())
            
            self.debug_print("Region data loaded successfully")
            
        except Exception as e:
            self.debug_print(f"Error loading region data: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())
            # Initialize empty DataFrames and dicts if loading fails
            self.region_data = pd.DataFrame(columns=['NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'CO_ENTIDADE'])
            self.uf_by_region = {}
            self.mun_by_uf = {}

    def get_results(self):
        """
        Obtém os resultados do banco de dados.
        """
        try:
            # Get current metadata
            current_metadata = self.query_metadata[
                (self.query_metadata['query_type'] == self.current_query_type) & 
                (self.query_metadata['query_model'] == self.current_query_model)
            ].iloc[0]
            
            group_by = str(current_metadata['group_by']).upper()
            self.debug_print(f"Query group by: {group_by}")
            
            has_entity = 'CO_ENTIDADE' in group_by
            self.debug_print(f"Has CO_ENTIDADE: {has_entity}")
            
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    if has_entity:
                        self.debug_print("Executing entity query...")
                        
                        # Get schools based on geographic filters
                        schools_filter = self.region_data.copy()
                        if self.region_dropdown.value != 'Todas':
                            schools_filter = schools_filter[
                                schools_filter['NO_REGIAO'] == self.region_dropdown.value
                            ]
                        if self.uf_dropdown.value != 'Todas':
                            schools_filter = schools_filter[
                                schools_filter['SG_UF'] == self.uf_dropdown.value
                            ]
                        if self.mun_dropdown.value != 'Todas':
                            schools_filter = schools_filter[
                                schools_filter['NO_MUNICIPIO'] == self.mun_dropdown.value
                            ]
                        
                        # Convert to strings since group_by_val1 is character varying
                        school_codes = [str(int(code)) for code in schools_filter['CO_ENTIDADE'].unique()]
                        if not school_codes:
                            self.debug_print("No schools found for the selected filters")
                            return pd.DataFrame()
                        
                        self.debug_print(f"Found {len(school_codes)} schools for the selected filters")
                        
                        # First, find the representative schools based on original values
                        find_schools_query = f"""
                            WITH base_values AS (
                                SELECT DISTINCT ON (group_by_val1) 
                                    group_by_val1,
                                    original_value
                                FROM {TABLE_NAMES['db_results_stats']}
                                WHERE query_type = %s 
                                AND query_model = %s
                                AND group_by_col1 = 'CO_ENTIDADE'
                                AND group_by_val1 = ANY(%s)
                            ),
                            stats AS (
                                SELECT *,
                                    AVG(original_value) OVER () as avg_val
                                FROM base_values
                            )
                            SELECT group_by_val1
                            FROM (
                                (SELECT group_by_val1 FROM stats ORDER BY original_value ASC LIMIT 1)
                                UNION ALL
                                (SELECT group_by_val1 
                                 FROM stats 
                                 ORDER BY ABS(original_value - avg_val) ASC 
                                 LIMIT 1)
                                UNION ALL
                                (SELECT group_by_val1 FROM stats ORDER BY original_value DESC LIMIT 1)
                            ) s;
                        """
                        
                        cur.execute(find_schools_query, (self.current_query_type, self.current_query_model, school_codes))
                        representative_schools = [row[0] for row in cur.fetchall()]
                        
                        self.debug_print(f"Selected representative schools: {representative_schools}")
                        
                        # Then, get all results for these schools
                        query = f"""
                            SELECT *
                            FROM {TABLE_NAMES['db_results_stats']}
                            WHERE query_type = %s 
                            AND query_model = %s
                            AND group_by_col1 = 'CO_ENTIDADE'
                            AND group_by_val1 = ANY(%s)
                            ORDER BY group_by_val1, epsilon, delta
                        """
                        
                        self.debug_print("Executing query for all privacy variations...")
                        cur.execute(query, (self.current_query_type, self.current_query_model, representative_schools))
                    
                    else:
                        self.debug_print("Executing standard query...")
                        # For query_type 1, don't apply geographic filters
                        if self.current_query_type == 1:
                            query = f"""
                                SELECT *
                                FROM {TABLE_NAMES['db_results_stats']}
                                WHERE query_type = %s
                                AND query_model = %s
                            """
                            cur.execute(query, (self.current_query_type, self.current_query_model))
                        else:
                            # Apply geographic filters for other query types
                            conditions, params = self.get_query_conditions()
                            
                            query = f"""
                                SELECT *
                                FROM {TABLE_NAMES['db_results_stats']}
                                WHERE {" AND ".join(conditions)}
                            """
                            self.debug_print(f"Executing query with conditions: {conditions}")
                            self.debug_print(f"Query parameters: {params}")
                            
                            cur.execute(query, params)
                    
                    columns = [desc[0] for desc in cur.description]
                    results = pd.DataFrame(cur.fetchall(), columns=columns)
                    
                    self.debug_print(f"Query returned {len(results)} rows")
                    return results
                    
        except Exception as e:
            self.debug_print(f"Database error: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())
            return pd.DataFrame()

    def get_query_conditions(self):
        """Get query conditions based on selected filters and metadata."""
        try:
            conditions = ['query_type = %s', 'query_model = %s']
            params = [self.current_query_type, self.current_query_model]
            
            # Get current metadata
            current_metadata = self.query_metadata[
                (self.query_metadata['query_type'] == self.current_query_type) & 
                (self.query_metadata['query_model'] == self.current_query_model)
            ].iloc[0]
            
            group_by = str(current_metadata['group_by']).upper()
            self.debug_print(f"Query group by: {group_by}")
            
            region = self.region_dropdown.value
            uf = self.uf_dropdown.value
            mun = self.mun_dropdown.value
            
            # Add conditions based on most specific selected filter
            if mun != 'Todas':
                conditions.append('UPPER(group_by_col1) = UPPER(%s)')
                conditions.append('UPPER(group_by_val1) = UPPER(%s)')
                params.extend(['NO_MUNICIPIO', mun])
            elif uf != 'Todas':
                conditions.append('UPPER(group_by_col1) = UPPER(%s)')
                conditions.append('UPPER(group_by_val1) = UPPER(%s)')
                params.extend(['SG_UF', uf])
            elif region != 'Todas':
                conditions.append('UPPER(group_by_col1) = UPPER(%s)')
                conditions.append('UPPER(group_by_val1) = UPPER(%s)')
                params.extend(['NO_REGIAO', region])
            
            # Remove None parameters and their corresponding conditions
            valid_conditions = []
            valid_params = []
            for cond, param in zip(conditions, params):
                if param is not None:
                    valid_conditions.append(cond)
                    valid_params.append(param)
            
            self.debug_print(f"Query conditions: {valid_conditions}")
            self.debug_print(f"Query parameters: {valid_params}")
            
            return valid_conditions, valid_params
            
        except Exception as e:
            self.debug_print(f"Error in get_query_conditions: {str(e)}")
            return [], []

    def _load_query_metadata(self):
        """
        Carrega os metadados das queries de um arquivo CSV.
        """
        try:
            # Get current notebook directory
            notebook_dir = Path().absolute()
            csv_file_path = notebook_dir / "queries_formatadas.csv"
            
            # Read the CSV file with proper encoding and separator
            self.query_metadata = pd.read_csv(
                csv_file_path,
                sep=';',
                encoding='utf-8'
            )
               
        except Exception as e:
            self.debug_print(f"Error loading query metadata: {str(e)}")
            # Initialize empty DataFrame if loading fails
            self.query_metadata = pd.DataFrame(columns=[
                'query_model', 'query_type', 'aggregation_type', 
                'aggregated_column', 'query_argument', 
                'aggregated_data', 'group_by'
            ]) 

    def update_privacy_params(self, results):
        """
        Update epsilon and delta dropdown options based on available values
        """
        if not results.empty:
            epsilon_values = sorted(results['epsilon'].unique())
            delta_values = sorted(results['delta'].unique())
            
            self.epsilon_dropdown.options = ['Todos'] + [str(eps) for eps in epsilon_values]
            self.delta_dropdown.options = ['Todos'] + [str(delta) for delta in delta_values]
        else:
            self.epsilon_dropdown.options = ['Todos']
            self.delta_dropdown.options = ['Todos']

    def _initial_plot(self):
        """
        Creates the initial plot with default values
        """
        try:
            # For query_type 1, we don't need region selection
            if self.current_query_type == 1:
                # Get results without geographic filters
                results = self.get_results()
                if results is not None and not results.empty:
                    self.update_combined_plot(results)
                    self.debug_print("Initial plot created successfully")
                else:
                    self.debug_print("No results available for initial plot")
            
        except Exception as e:
            self.debug_print(f"Error in initial plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc()) 

    def update_stats_plot(self, results, selected_stat):
        """Atualiza o gráfico de estatísticas usando os metadados da query."""
        try:
            with self.stats_fig_widget.batch_update():
                self.stats_fig_widget.data = []
                
                if results.empty:
                    self.debug_print("No results to plot in stats chart")
                    return
                
                # Debug information
                self.debug_print("\nUpdating stats plot:")
                self.debug_print(f"Selected stat: {selected_stat}")
                self.debug_print(f"Number of rows: {len(results)}")
                
                # Sort results by original_value
                results = results.sort_values('original_value', ascending=True)
                
                # Create labels for x-axis
                x_labels = []
                for _, row in results.iterrows():
                    label_parts = []
                    for i in range(1, 4):
                        col = f'group_by_col{i}'
                        val = f'group_by_val{i}'
                        if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ['', 'None']:
                            label_parts.append(f"{str(row[col]).strip()}: {str(row[val]).strip()}")
                    x_labels.append('\n'.join(label_parts) if label_parts else 'N/A')
                
                # Handle different plot types
                if selected_stat == 'ci':
                    # Plot confidence interval as a range
                    y_values = results['dp_avg'].fillna(0).astype(float)
                    ci_lower = results['dp_ci_lower'].fillna(0).astype(float)
                    ci_upper = results['dp_ci_upper'].fillna(0).astype(float)
                    
                    self.debug_print(f"Plotting CI values:")
                    self.debug_print(f"Mean: {y_values.tolist()}")
                    self.debug_print(f"Lower: {ci_lower.tolist()}")
                    self.debug_print(f"Upper: {ci_upper.tolist()}")
                    
                    # Add mean line
                    self.stats_fig_widget.add_trace(
                        go.Scatter(
                            name='Média',
                            x=x_labels,
                            y=y_values,
                            mode='lines+markers',
                            line=dict(color='rgb(55, 83, 109)', width=2),
                            marker=dict(size=8)
                        )
                    )
                    
                    # Add confidence interval
                    self.stats_fig_widget.add_trace(
                        go.Scatter(
                            name='Intervalo de Confiança',
                            x=x_labels + x_labels[::-1],  # x, then x reversed
                            y=ci_upper.tolist() + ci_lower.tolist()[::-1],  # upper, then lower reversed
                            fill='toself',
                            fillcolor='rgba(55, 83, 109, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
                else:
                    # Regular bar plot for other statistics
                    stat_columns = {
                        'mae': 'mae',
                        'mape': 'mape',
                        'dp_avg': 'dp_avg',
                        'dp_median': 'dp_median',
                        'dp_stddev': 'dp_stddev',
                        'original': 'original_value'
                    }
                    
                    if selected_stat not in stat_columns:
                        self.debug_print(f"Error: Unknown statistic {selected_stat}")
                        return
                        
                    y_values = results[stat_columns[selected_stat]].fillna(0).astype(float)
                    self.debug_print(f"Plotting {selected_stat} values: {y_values.tolist()}")
                    
                    self.stats_fig_widget.add_trace(
                        go.Bar(
                            name=selected_stat,
                            x=x_labels,
                            y=y_values,
                            text=[f'{v:,.4f}' if v != 0 else 'N/A' for v in y_values],
                            textposition='outside',
                            textangle=-45,
                            marker_color='rgb(55, 83, 109)'
                        )
                    )
                
                # Update layout
                self.stats_fig_widget.update_layout(
                    title=dict(
                        text='Intervalo de Confiança' if selected_stat == 'ci' else f'Estatísticas para {selected_stat}',
                        y=0.95,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    xaxis_title="Grupos",
                    yaxis_title="Valor",
                    showlegend=True,
                    xaxis_tickangle=-45,
                    height=500,
                    margin=dict(b=150)  # Add more bottom margin for labels
                )
                
        except Exception as e:
            self.debug_print(f"Error updating stats plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

        # Update stats dropdown options
        self.stats_dropdown = widgets.Dropdown(
            options=[
                ('MAE', 'mae'),
                ('MAPE', 'mape'),
                ('Média DP', 'dp_avg'),
                ('Mediana DP', 'dp_median'),
                ('Desvio Padrão DP', 'dp_stddev'),
                ('Valor Original', 'original'),
                ('Intervalo de Confiança', 'ci')  # Added CI option
            ],
            value='dp_avg',
            description='Estatística:'
        ) 
        