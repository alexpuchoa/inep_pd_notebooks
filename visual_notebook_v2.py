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
import logging
from pathlib import Path
import plotly.io as pio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# This line might have been removed or changed
pio.renderers.default = 'colab'

class VisualizationNotebook:
    """
    Classe principal para visualização de dados com Differential Privacy.
    Gerencia a interface interativa e a exibição de gráficos comparativos.
    """

    def __init__(self, data=None, queries_config=None, data_path=None):
        """
        Inicializa o notebook de visualização.
        """
        try:
            print("Carregando App")
            
            # Create debug output widget first
            self.debug_output = widgets.Output()
            print("Debug widget created")
            display(self.debug_output)
            
            # Initialize current values
            self.current_query_type = 1  # Add this line if it's missing
            print(f"Initial query type: {self.current_query_type}")
            
            with self.debug_output:
                print("Debug output is working")
            
            # Set data path
            self.data_path = Path(data_path)            

            if data is not None and queries_config is not None:
                # Use provided data
                self.df = data
                self.queries_config = queries_config
                print(f"Using provided data. Shape: {self.df.shape}")
            
            elif data_path:
                # Load data from files

                csv_path = self.data_path / "dp_results_stats_bq.csv"
                queries_file = self.data_path / "queries_formatadas_bq.csv"
                
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
                        'original_value': 'float64',
                        'group_by_val1': 'str'
                    },
                    sep=';',
                    encoding='latin1',
                    low_memory=False,
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                
                self.df = pd.concat(chunks, ignore_index=True)
                print(f"Data loaded successfully. Shape: {self.df.shape}")
                
                # Load queries configuration
                self.queries_config = pd.read_csv(queries_file, sep=';')
            
            else:
                raise ValueError("Either data and queries_config or data_path must be provided")
            
            # Create other widgets and connect observers
            self._create_widgets()
            
            # Define the initial aggregation model for the query
            self.query_model_aggregation = {1: 'Soma Alunos', 2: 'Média Nota', 3: 'Soma Nota'}
            self.selected_model = self.query_model_aggregation[1]

            self.region_data = None
            self.uf_by_region = {}
            self.mun_by_uf = {}
            
            with self.debug_output:
                print("DEBUG: Initializing VisualizationNotebook")
            
            # Load region data
            self._load_region_data()
            
            # Connect observers
            self._connect_observers()
            
            # Display interface
            self.display_stats_chart()
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _load_data(self):
        """Load data from CSV files."""
        try:
            # Load results data
            self.df = pd.read_csv(self.results_path, sep=';', encoding='latin1', low_memory=False)
            
            # Load queries configuration
            self.query_metadata = pd.read_csv(self.queries_file, sep=';')

            with self.debug_output:
                print(f"Data loaded successfully")
                print(f"Results shape: {self.df.shape}")
                print(f"Query metadata shape: {self.query_metadata.shape}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

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
            # Only connect geographic filters for cascading dropdowns
            self.region_dropdown.observe(self._on_region_change, names='value')
            self.uf_dropdown.observe(self._on_uf_change, names='value')
            self.mun_dropdown.observe(self._on_mun_change, names='value')
            
            # Connect submit button - all parameter updates happen here
            with self.debug_output:
                print("Connecting button...")
            self.submit_button.on_click(self.update_both_plots)
            with self.debug_output:
                print("Button connected!")
            
        except Exception as e:
            self.debug_print(f"ERROR connecting observers: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _on_query_type_change(self, change):
        """
        Handler for query type changes.
        """
        self.current_query_type = change.new
        print(f"Query type changed to: {self.current_query_type}")

    def _on_query_model_change(self, change):
        """Handler específico para mudanças no query_model"""
        #self.debug_print(f"\nDEBUG: Query model changed to {change.new}")
        self.update_both_plots(None)

    def _on_region_change(self, change):
        """Handler para mudanças na região"""
        with self.debug_output:
            print(f"\nRegião alterada para: {change.new}")
            
        if change.new == 'Todas':
            self.uf_dropdown.options = ['Todas'] + sorted(self.region_data['SG_UF'].unique().tolist())
        else:
            self.uf_dropdown.options = ['Todas'] + sorted(self.uf_by_region[change.new])
            self.uf_dropdown.value = 'Todas'
            self.mun_dropdown.options = ['Todas']
            self.mun_dropdown.value = 'Todas'

    def _on_uf_change(self, change):
        """Handler para mudanças na UF"""
        with self.debug_output:
            print(f"\nUF alterada para: {change.new}")
            
        if change.new == 'Todas':
            region = self.region_dropdown.value
            if region != 'Todas':
                municipalities = sorted(self.region_data[
                    self.region_data['NO_REGIAO'] == region
                ]['NO_MUNICIPIO'].unique().tolist())
            else:
                municipalities = sorted(self.region_data['NO_MUNICIPIO'].unique().tolist())
            self.mun_dropdown.options = ['Todas'] + municipalities
        else:
            self.mun_dropdown.options = ['Todas'] + sorted(self.mun_by_uf[change.new])
        self.mun_dropdown.value = 'Todas'

    def _on_mun_change(self, change):
        """Handler para mudanças no município"""
        with self.debug_output:
            print(f"\nMunicípio alterado para: {change.new}")
        self.update_both_plots(None)  # Force update after municipality change

    def _on_parameter_change(self, change):
        """Handler para mudanças nos parâmetros epsilon/delta/stats"""
        #self.debug_print(f"\nDEBUG: Parameter {change.owner.description} changed to {change.new}")
        self.update_both_plots(None)

    def display_stats_chart(self):
        """
        Display the visualization interface with improved layout
        """
        try:
            # Title with professional styling
            title = widgets.VBox([
                widgets.HTML("<h2>Métricas das 25 execuções da PD</h2><h3>Dependendo da seleção, o eixo-x pode não comportar o nome de todas as entidades.</h3>"),
                #widgets.HTML("<p>Filtros de Granularidade Geográfica - Seleção obrigatória para as queries com maior granularidade (5, 6, 7, 8).</p>")
            ])
            

            # Group controls with labels
            query_controls = widgets.VBox([
                widgets.HTML("<b>Tipo e Modelo da Query (ver tabela acima)</b>"),
                widgets.HBox([
                    self.query_type_dropdown,
                    self.query_model_dropdown
                ])
            ])
            
            parameter_controls = widgets.VBox([
                widgets.HTML("<b>Parâmetros da PD</b>"),
                widgets.HBox([
                    self.epsilon_dropdown,
                    self.delta_dropdown
                ])
            ])
            
            geographic_controls = widgets.VBox([
                widgets.HTML("<b>Filtros de Granularidade Geográfica</b>"),
                widgets.HBox([
                    self.region_dropdown,
                    self.uf_dropdown,
                    self.mun_dropdown
                ])
            ])
            
            stats_control = widgets.VBox([
                widgets.HTML("<b>Estatísticas</b>"),
                self.stats_dropdown
            ])
            
            # Main container with all controls
            controls = widgets.VBox([
                query_controls,
                parameter_controls,
                geographic_controls,
                stats_control,
                self.submit_button
            ])
            
            # Plots container
            plots = widgets.VBox([
                self.stats_fig_widget,
                self.bars_fig_widget
            ])
            
            # Main container
            self.stats_container = widgets.VBox([
                title,
                controls,
                plots
            ])
            
            # Display the interface
            display(self.stats_container)
            
            # Initialize empty plots
            with self.stats_fig_widget.batch_update():
                self.stats_fig_widget.data = []
            with self.bars_fig_widget.batch_update():
                self.bars_fig_widget.data = []
            
        except Exception as e:
            print(f"Error displaying interface: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def _create_widgets(self):
        """
        Cria todos os widgets da interface.
        """
        # Query type dropdown
        self.query_type_dropdown = widgets.Dropdown(
            options=[(f'Query {i}', i) for i in range(1, 9)],
            value=1,
            description='Query Type:',
            style={'description_width': 'initial'}
        )
        
        # Query model dropdown
        self.query_model_dropdown = widgets.Dropdown(
            options=[1, 2, 3],
            value=1,
            description='Query Model:'
        )
        
        # Region dropdown
        self.region_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Região:'
        )
        
        # UF dropdown
        self.uf_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='UF:'
        )
        
        # Municipality dropdown
        self.mun_dropdown = widgets.Dropdown(
            options=['Todas'],
            value='Todas',
            description='Município:'
        )
        
        # Epsilon dropdown
        self.epsilon_dropdown = widgets.Dropdown(
            options=[1.0, 5.0, 10.0],
            value=1.0,
            description='Epsilon:'
        )
        
        # Delta dropdown
        self.delta_dropdown = widgets.Dropdown(
            options=[1e-5, 1e-2],
            value=1e-5,
            description='Delta:'
        )
        
        # Stats dropdown
        self.stats_dropdown = widgets.Dropdown(
            options=[
                ('MAE', 'mae'),
                ('MAPE', 'mape'),
                ('Mediana DP', 'dp_median'),
                ('Desvio Padrão DP', 'dp_stddev'),
                ('Intervalo de Confiança', 'ci')
            ],
            value='mae',
            description='Estatística:'
        )
        
        # Submit button
        self.submit_button = widgets.Button(
            description='Atualizar Gráficos',
            button_style='primary',
            tooltip='Clique para atualizar os gráficos com as seleções atuais'
        )
        
        # Debug output with increased height
        self.debug_output = widgets.Output(
            layout=widgets.Layout(
                height='200px',
                border='1px solid black',
                overflow_y='auto'
            )
        )
        
        # Create empty figures with consistent size
        plot_width = 1000  # Increased width
        plot_height = 500
        
        self.stats_fig_widget = go.FigureWidget(
            layout=go.Layout(
                height=plot_height,
                width=plot_width,
                margin=dict(t=50, b=100, l=100, r=50)
            )
        )
        
        self.bars_fig_widget = go.FigureWidget(
            layout=go.Layout(
                height=plot_height,
                width=plot_width,
                margin=dict(t=50, b=100, l=100, r=50)
            )
        )

    def _load_region_data(self):
        """
        Carrega dados geográficos do arquivo CSV.
        """
        try:
            # Get current notebook directory
            #csv_file_path = self.data_path / "regiao_uf_municipio_escola.csv"
            self.region_data = pd.read_csv(self.data_path / "regiao_uf_municipio_escola.csv", sep=';')

            # Debug: Show sample of the CSV data
            self.debug_print("\nSample of CSV data:")
            self.debug_print("\nColumns:")
            self.debug_print(self.region_data.columns.tolist())
            self.debug_print("\nFirst 5 rows:")
            self.debug_print(self.region_data.head().to_string())
            
            # Rest of the function remains the same
            unique_regions = sorted(self.region_data['NO_REGIAO'].unique())
            self.region_dropdown.options = ['Todas'] + unique_regions
            
            # Prepara os dicionários de relação
            self.uf_by_region = {}
            self.mun_by_uf = {}
            
            for region in unique_regions:
                region_data = self.region_data[self.region_data['NO_REGIAO'] == region]
                self.uf_by_region[region] = sorted(region_data['SG_UF'].unique())
                
                for uf in self.uf_by_region[region]:
                    uf_data = region_data[region_data['SG_UF'] == uf]
                    self.mun_by_uf[uf] = sorted(uf_data['NO_MUNICIPIO'].unique())
            
            # Atualiza UF dropdown com todas as UFs
            all_ufs = sorted(self.region_data['SG_UF'].unique())
            self.uf_dropdown.options = ['Todas'] + all_ufs
            
            # Build hierarchical relationships from the CSV file
            self.debug_print("\nBuilding hierarchical relationships from CSV...")
            region_to_ufs = {}
            uf_to_muns = {}
            mun_to_schools = {}
            
            # First pass: Build all relationships from CSV
            for _, row in self.region_data.iterrows():
                # Force uppercase for all fields
                region = row['NO_REGIAO'].strip().upper()
                uf = row['SG_UF'].strip().upper()
                mun = row['NO_MUNICIPIO'].strip().upper()
                escola = str(row['CO_ENTIDADE']).strip().upper()
                
                # Build region -> UF relationship
                if region not in region_to_ufs:
                    region_to_ufs[region] = set()
                region_to_ufs[region].add(uf)
                
                # Build UF -> Municipality relationship
                if uf not in uf_to_muns:
                    uf_to_muns[uf] = set()
                uf_to_muns[uf].add(mun)
                
                # Build Municipality -> School relationship
                if mun not in mun_to_schools:
                    mun_to_schools[mun] = set()
                mun_to_schools[mun].add(escola)
            
            # Create reverse lookups
            uf_to_region = {uf: region for region, ufs in region_to_ufs.items() for uf in ufs}
            mun_to_uf = {mun: uf for uf, muns in uf_to_muns.items() for mun in muns}
            
            # Debug information
            self.debug_print("\nSample of relationships:")
            self.debug_print("\nRegion -> UFs:")
            for region in list(region_to_ufs.keys())[:2]:
                self.debug_print(f"{region}: {sorted(region_to_ufs[region])}")
            
            self.debug_print("\nUF -> Municipalities (first 2 UFs):")
            for uf in list(uf_to_muns.keys())[:2]:
                self.debug_print(f"{uf}: {sorted(list(uf_to_muns[uf]))[:3]}...")
            
            self.debug_print("\nMunicipality -> Schools (first 2 municipalities):")
            for mun in list(mun_to_schools.keys())[:2]:
                self.debug_print(f"{mun}: {sorted(list(mun_to_schools[mun]))[:3]}...")
            
            self.debug_print(f"""
                Hierarchical relationships built from CSV:
                - {len(region_to_ufs)} regions
                - {len(uf_to_muns)} UFs
                - {len(mun_to_schools)} municipalities
                - {sum(len(schools) for schools in mun_to_schools.values()):,} total schools
            """)
            
        except Exception as e:
            print(f"Error loading region data: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def update_both_plots(self, button_clicked=None):
        print("Button clicked - updating plots")
        print(f"Raw dropdown value: {self.query_type_dropdown.value}")
        try:
            # Use dropdown value instead of slider
            self.current_query_type = self.query_type_dropdown.value
            print(f"current_query_type after assignment: {self.current_query_type}")
            
            # Rest of the updates
            self.current_query_model = self.query_model_dropdown.value
            self.current_epsilon = self.epsilon_dropdown.value
            self.current_delta = self.delta_dropdown.value
            self.current_stat = self.stats_dropdown.value
            
            print(f"Parameters updated on button click:")
            print(f"Query Type: {self.current_query_type}")
            print(f"Query Model: {self.current_query_model}")
            print(f"Epsilon: {self.current_epsilon}")
            print(f"Delta: {self.current_delta}")
            print(f"Stat: {self.current_stat}")
            
            # Get filtered results
            filtered_results = self.df[
                (self.df['query_type'] == self.current_query_type) &
                (self.df['query_model'] == self.current_query_model) &
                (self.df['epsilon'] == self.current_epsilon) &
                (self.df['delta'] == self.current_delta)
            ]
            
            with self.debug_output:
                print(f"Initial filter returned {len(filtered_results)} rows")
                print(f"Sample of filtered data:")
                if not filtered_results.empty:
                    print(filtered_results[['query_type', 'query_model', 'epsilon', 'delta', self.current_stat]].head())
            
            # Add debug prints
            print("\nChecking query configuration:")
            print(f"Query type: {self.current_query_type}")
            query_metadata = self.queries_config[
                (self.queries_config['query_type'] == self.current_query_type)
            ]
            print("Query metadata:")
            print(query_metadata[['query_type', 'group_by', 'aggregation_type']].to_string())
            
            # After filtering
            print("\nFiltered data group by columns:")
            if not filtered_results.empty:
                for i in range(1, 4):
                    col = f'group_by_col{i}'
                    if col in filtered_results.columns:
                        print(f"{col}: {filtered_results[col].unique()}")
            
            if not filtered_results.empty:
                # Apply geographic filters
                filtered_results = self.filter_results(
                    filtered_results, 
                    self.region_dropdown.value,
                    self.uf_dropdown.value,
                    self.mun_dropdown.value
                )
                
                with self.debug_output:
                    print(f"After geographic filter: {len(filtered_results)} rows")
                    print(f"Sample after geographic filter:")
                    if not filtered_results.empty:
                        print(filtered_results[['parent_regiao', 'parent_uf', 'parent_municipio', self.current_stat]].head())
                
                # Right before updating plots
                print(f"About to update plots with {len(filtered_results)} rows")
                
                # Update plots
                self.update_stats_plot(filtered_results, self.current_stat)
                print("Stats plot updated")
                
                self.update_bars_plot(filtered_results)
                print("Bars plot updated")
                
                with self.debug_output:
                    print("Plots updated successfully")
                
            else:
                with self.debug_output:
                    print("No data found for selected parameters")
            
        except Exception as e:
            with self.debug_output:
                print(f"Error updating plots: {str(e)}")
                import traceback
                print(traceback.format_exc())

    def filter_results(self, results, region='Todas', uf='Todas', mun='Todas'):
        """Filtra os resultados com base nas seleções geográficas."""
        try:
            if results is None or results.empty:
                return results
            
            filtered_results = results.copy()
            
            # Convert all string columns to string type first
            for col in ['parent_regiao', 'parent_uf', 'parent_municipio', 'group_by_val1']:
                if col in filtered_results.columns:
                    filtered_results[col] = filtered_results[col].astype(str)
            
            # Convert filter values to strings
            region = str(region).strip().upper() if region != 'Todas' else region
            uf = str(uf).strip().upper() if uf != 'Todas' else uf
            mun = str(mun).strip().upper() if mun != 'Todas' else mun
            
            print(f"\nFiltering for:")
            print(f"Region: {region}")
            print(f"UF: {uf}")
            print(f"Municipality: {mun}")
            
            # Apply filters
            if region != 'Todas':
                filtered_results = filtered_results[filtered_results['parent_regiao'].str.upper() == region]
                print(f"After region filter: {len(filtered_results)} rows")
            
            if uf != 'Todas':
                filtered_results = filtered_results[filtered_results['parent_uf'].str.upper() == uf]
                print(f"After UF filter: {len(filtered_results)} rows")
            
            if mun != 'Todas':
                filtered_results = filtered_results[filtered_results['parent_municipio'].str.upper() == mun]
                print(f"After municipality filter: {len(filtered_results)} rows")
            
            print("\nUnique values in group_by_val1:")
            print(filtered_results['group_by_val1'].unique()[:5])
            
            return filtered_results

        except Exception as e:
            print(f"Error in filtering: {str(e)}")
            return results

    def update_stats_plot(self, results, selected_stat):
        """Atualiza o gráfico de estatísticas usando os metadados da query."""
        try:
            with self.stats_fig_widget.batch_update():
                self.stats_fig_widget.data = []
                
                if results.empty:
                    self.debug_print("No results to plot in stats chart")
                    return
                
                # Get query metadata for current selection
                current_metadata = self.queries_config[
                    (self.queries_config['query_model'] == self.current_query_model) & 
                    (self.queries_config['query_type'] == self.current_query_type)
                ].iloc[0]
                
                # Get aggregation type for y-axis title
                aggregation_type = current_metadata['aggregation_type']
                
                # Get all group by columns for x-axis title
                group_by_cols = []
                for i in range(1, 4):  # Check all possible group_by columns
                    col = f'group_by_col{i}'
                    if col in results.columns and pd.notna(results[col].iloc[0]):
                        col_value = str(results[col].iloc[0]).strip()
                        if col_value:
                            if col_value == 'NO_REGIAO': group_by_cols.append('Região')
                            elif col_value == 'SG_UF': group_by_cols.append('Estado')
                            elif col_value == 'NO_MUNICIPIO': group_by_cols.append('Município')
                
                x_axis_title = ' / '.join(group_by_cols) if group_by_cols else 'Agrupamento'
                
                # Sort results by original_value to maintain consistent order with bars plot
                results = results.sort_values('original_value', ascending=True)
                
                # Get full Portuguese names for statistics
                stat_names = {
                    'mae': 'Erro Médio Absoluto (MAE)',
                    'mape': 'Erro Médio Absoluto Percentual (MAPE)',
                    'dp_avg': 'Média',
                    'dp_median': 'Mediana',
                    'dp_stddev': 'Desvio Padrão',
                    'original': 'Valor Original',
                    'ci': 'Intervalo de Confiança'
                }
                
                # Get geographic grouping columns being shown
                group_cols = []
                for i in range(1, 4):
                    col = f'group_by_col{i}'
                    if col in results.columns and pd.notna(results[col].iloc[0]):
                        col_value = str(results[col].iloc[0]).strip()
                        if col_value:
                            if col_value == 'NO_REGIAO': group_cols.append('Região')
                            elif col_value == 'SG_UF': group_cols.append('Estado')
                            elif col_value == 'NO_MUNICIPIO': group_cols.append('Município')
                
                # Get geographic values being shown
                group_vals = []
                for i in range(1, 4):
                    val = f'group_by_val{i}'
                    if val in results.columns and pd.notna(results[val].iloc[0]):
                        val_str = str(results[val].iloc[0]).strip()
                        if val_str:
                            group_vals.append(val_str)
                
                # Create descriptive title
                title = f"{stat_names.get(selected_stat, selected_stat)}\n"
                title += f"por {' e '.join(group_cols)}"
                #if group_vals:
                #    title += f": {' - '.join(group_vals)}"
                
                # Create x-axis labels
                x_labels = []
                for _, row in results.iterrows():
                    parts = []
                    for i in range(1, 4):
                        val = f'group_by_val{i}'
                        if val in row and pd.notna(row[val]) and str(row[val]).strip():
                            parts.append(str(row[val]).strip())
                    x_labels.append('\n'.join(parts) if parts else 'N/A')
                
                # Handle different plot types
                if selected_stat == 'ci':
                    y_values = results['dp_avg'].fillna(0).astype(float)
                    ci_lower = results['dp_ci_lower'].fillna(0).astype(float)
                    ci_upper = results['dp_ci_upper'].fillna(0).astype(float)
                    
                    # Calculate interval sizes
                    interval_sizes = ci_upper - ci_lower
                    
                    # Handle single entity case
                    if len(x_labels) == 1:
                        # Duplicate the point to create a visible line/area
                        x_labels = x_labels + [x_labels[0] + ' ']  # Add space to create distinct label
                        interval_sizes = pd.concat([interval_sizes, interval_sizes])
                    
                    # Add interval shadows
                    self.stats_fig_widget.add_trace(
                        go.Scatter(
                            name='Intervalos',
                            x=x_labels,
                            y=interval_sizes,
                            mode='none',
                            fill='tonexty',
                            fillcolor='rgba(55, 83, 109, 0.15)',  # Lighter shadow
                            showlegend=False
                        )
                    )
                    
                    # Add just the line for interval sizes
                    self.stats_fig_widget.add_trace(
                        go.Scatter(
                            name='Intervalo',
                            x=x_labels,
                            y=interval_sizes,
                            mode='lines',  # Only lines, no markers or text
                            line=dict(
                                color='rgb(55, 83, 109)', 
                                width=1.5  # Thinner line
                            ),
                            connectgaps=False
                        )
                    )
                    
                    # Update y-axis range based on selected statistic
                    if selected_stat == 'ci':
                        y_max = interval_sizes.max()
                        y_min = 0
                        y_title = 'Tamanho do Intervalo de Confiança'
                    else:
                        y_values = results[stat_columns[selected_stat]].fillna(0).astype(float)
                        y_max = y_values.max()
                        y_min = y_values.min()
                        y_title = stat_names[selected_stat]
                    
                    # Add padding to the range
                    y_range = y_max - y_min
                    y_max += y_range * 0.1  # 10% padding on top
                    if y_min > 0:
                        y_min = max(0, y_min - y_range * 0.1)  # 10% padding on bottom, but not below 0
                    
                    self.stats_fig_widget.update_layout(
                        yaxis=dict(
                            range=[y_min, y_max],
                            title=y_title
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
                    
                    self.stats_fig_widget.add_trace(
                        go.Bar(
                            name=stat_names[selected_stat],
                            x=x_labels,
                            y=y_values,
                            text=[f'{v:,.4f}' for v in y_values],
                            textposition='outside',
                            marker_color='rgb(55, 83, 109)'
                        )
                    )
                
                # Update layout with adjusted title position and margins
                self.stats_fig_widget.update_layout(
                    title=dict(
                        text=title,
                        y=0.95,  # Reduced from 0.98
                        x=0.5,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=14)
                    ),
                    xaxis_title=dict(
                        text=x_axis_title,
                        font=dict(size=12)
                    ),
                    yaxis_title=dict(
                        text=aggregation_type,
                        font=dict(size=12)
                    ),
                    showlegend=False,
                    xaxis_tickangle=-45,
                    height=600,
                    width=1000,
                    margin=dict(
                        t=100,  # Reduced from 150
                        b=100,
                        l=100,
                        r=50
                    )
                )
                
        except Exception as e:
            self.debug_print(f"Error updating stats plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def update_bars_plot(self, results):
        """Atualiza o gráfico de barras usando os metadados da query."""
        try:
            with self.bars_fig_widget.batch_update():
                self.bars_fig_widget.data = []
                
                if results.empty:
                    return
                
                # Get query metadata for current selection
                current_metadata = self.queries_config[
                    (self.queries_config['query_model'] == self.current_query_model) & 
                    (self.queries_config['query_type'] == self.current_query_type)
                ].iloc[0]
                
                # Get aggregation type and data for axis titles
                aggregation_type = current_metadata['aggregation_type']
                group_by = current_metadata['group_by']
                
                # Sort results by original_value
                results = results.sort_values('original_value', ascending=True)
                
                # Create labels for x-axis (maintaining sort order)
                x_labels = []
                for _, row in results.iterrows():
                    label_parts = []
                    for i in range(1, 4):
                        col = f'group_by_col{i}'
                        val = f'group_by_val{i}'
                        
                        if col in row and pd.notna(row[col]) and str(row[col]) not in ['None', '']:
                            val_str = str(row[val])
                            if '.' in val_str:
                                val_str = val_str.split('.')[0]
                            label_parts.append(val_str)
                            
                    x_labels.append(' | '.join(label_parts) if label_parts else 'N/A')
                
                # Convert values to float and handle None values
                original_values = results['original_value'].fillna(0).astype(float)
                dp_values = results['dp_avg'].fillna(0).astype(float)
                
                # Add traces with professional colors
                self.bars_fig_widget.add_trace(
                    go.Bar(
                        name='Valor Original',
                        x=x_labels,
                        y=original_values,
                        text=[f'{v:,.2f}' if pd.notna(v) else 'N/A' for v in original_values],
                        textposition='outside',
                        textangle=-45,
                        marker_color='rgb(55, 83, 109)'
                    )
                )
                
                self.bars_fig_widget.add_trace(
                    go.Bar(
                        name='Média PD',
                        x=x_labels,
                        y=dp_values,
                        text=[f'{v:,.2f}' if pd.notna(v) else 'N/A' for v in dp_values],
                        textposition='outside',
                        textangle=-45,
                        marker_color='rgb(26, 118, 255)'
                    )
                )
                
                # Get max value considering both original and dp values
                max_value = max(
                    original_values.max(),
                    dp_values.max()
                )
                
                # Update layout with adjusted title position and margins
                self.bars_fig_widget.update_layout(
                    title=dict(
                        text=f'Comparação: {aggregation_type} Original vs Média PD',
                        y=0.95,  # Matches upper plot
                        x=0.5,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=14)
                    ),
                    xaxis_title=dict(
                        text=group_by,
                        font=dict(size=12)
                    ),
                    yaxis_title=dict(
                        text=aggregation_type,
                        font=dict(size=12)
                    ),
                    yaxis=dict(
                        range=[0, max_value * 1.4],
                        title=dict(
                            text=aggregation_type,
                            font=dict(size=12)
                        )
                    ),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.95,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1
                    ),
                    xaxis_tickangle=-45,
                    barmode='group',
                    margin=dict(
                        t=100,  # Reduced from 150
                        b=100,
                        l=100,
                        r=50
                    ),
                    width=1000,
                    height=600
                )
                
        except Exception as e:
            self.debug_print(f"Error updating bars plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def get_results(self, query_type, query_model, epsilon, delta):
        """
        Obtém os resultados do banco de dados usando cursor.fetchall().
        """
        try:
            # Debug print
            self.debug_print(f"""
                Fetching results for:
                - Query Type: {query_type}
                - Query Model: {query_model}
                - Epsilon: {epsilon}
                - Delta: {delta}
            """)
            
            # Connect and create cursor
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            
            query = f"""
                SELECT *
                FROM dp_results_stats_bq
                WHERE query_type = %s
                AND query_model = %s
                AND epsilon = %s
                AND delta = %s;
            """
            
            # Execute query
            cur.execute(query, (query_type, query_model, epsilon, delta))
            
            # Get column names
            columns = [desc[0] for desc in cur.description]
            
            # Fetch all results
            rows = cur.fetchall()
            
            # Debug print
            self.debug_print(f"Found {len(rows)} results")
            
            if not rows:
                # Try to find what combinations exist
                cur.execute("""
                    SELECT DISTINCT query_type, query_model, epsilon, delta 
                    FROM dp_results_stats_bq 
                    ORDER BY query_type, query_model, epsilon, delta
                """)
                available = cur.fetchall()
                self.debug_print("Available combinations:")
                for combo in available:
                    self.debug_print(f"type={combo[0]}, model={combo[1]}, ε={combo[2]}, δ={combo[3]}")
            
            # Create DataFrame
            results = pd.DataFrame(rows, columns=columns)
            
            # Debug print column names and first row if exists
            if not results.empty:
                self.debug_print("\nColumns:")
                self.debug_print(results.columns.tolist())
                self.debug_print("\nFirst row:")
                self.debug_print(results.iloc[0].to_dict())
            
            return results
            
        except Exception as e:
            self.debug_print(f"Database error: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    def update_geographic_hierarchy(self, force_update=False):
        """Updates dp_results_stats_bq with parent geographic columns."""
        try:
            self.debug_print("Starting geographic hierarchy update...")
            
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Initial statistics query
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(parent_regiao) as with_region,
                            COUNT(parent_uf) as with_uf,
                            COUNT(parent_municipio) as with_mun,
                            COUNT(*) FILTER (WHERE group_by_col1 = 'NO_REGIAO') as total_regions,
                            COUNT(*) FILTER (WHERE group_by_col1 = 'SG_UF') as total_ufs,
                            COUNT(*) FILTER (WHERE group_by_col1 = 'NO_MUNICIPIO') as total_municipalities,
                            COUNT(*) FILTER (WHERE group_by_col1 = 'CO_ENTIDADE') as total_schools,
                            COUNT(*) FILTER (WHERE group_by_col1 = 'CO_ENTIDADE' AND parent_municipio IS NULL) as pending_schools
                        FROM dp_results_stats_bq;
                    """)
                    stats = cur.fetchone()
                    
                    (total_rows, with_region, with_uf, with_mun,
                     total_regions, total_ufs, total_municipalities, total_schools,
                     pending_schools) = stats
                    
                    self.debug_print(f"""
                        Current database statistics:
                        
                        Total rows: {total_rows:,}
                        
                        Rows by geographic level:
                        - Regions: {total_regions:,}
                        - UFs: {total_ufs:,}
                        - Municipalities: {total_municipalities:,}
                        - Schools: {total_schools:,}
                        
                        Parent column status:
                        - With region: {with_region:,} ({with_region/total_rows*100:.1f}%)
                        - With UF: {with_uf:,} ({with_uf/total_rows*100:.1f}%)
                        - With municipality: {with_mun:,} ({with_mun/total_rows*100:.1f}%)
                        
                        Pending updates:
                        - Schools needing municipality info: {pending_schools:,}
                        - Estimated batches (1000 schools per batch): {pending_schools // 1000 + 1:,}
                    """)
                    
                    # Build hierarchical relationships from CSV
                    self.debug_print("\nBuilding hierarchical relationships from CSV...")
                    region_to_ufs = {}
                    uf_to_muns = {}
                    mun_to_schools = {}
                    
                    # First pass: Build all relationships from CSV
                    for _, row in self.region_data.iterrows():
                        # Force uppercase for all fields
                        region = row['NO_REGIAO'].strip().upper()
                        uf = row['SG_UF'].strip().upper()
                        mun = row['NO_MUNICIPIO'].strip().upper()
                        escola = str(row['CO_ENTIDADE']).strip().upper()
                        
                        # Build region -> UF relationship
                        if region not in region_to_ufs:
                            region_to_ufs[region] = set()
                        region_to_ufs[region].add(uf)
                        
                        # Build UF -> Municipality relationship
                        if uf not in uf_to_muns:
                            uf_to_muns[uf] = set()
                        uf_to_muns[uf].add(mun)
                        
                        # Build Municipality -> School relationship
                        if mun not in mun_to_schools:
                            mun_to_schools[mun] = set()
                        mun_to_schools[mun].add(escola)
                    
                    # Create reverse lookups
                    uf_to_region = {uf: region for region, ufs in region_to_ufs.items() for uf in ufs}
                    mun_to_uf = {mun: uf for uf, muns in uf_to_muns.items() for mun in muns}
                    
                    # Debug information
                    self.debug_print("\nSample of relationships:")
                    self.debug_print("\nRegion -> UFs:")
                    for region in list(region_to_ufs.keys())[:2]:
                        self.debug_print(f"{region}: {sorted(region_to_ufs[region])}")
                    
                    self.debug_print("\nUF -> Municipalities (first 2 UFs):")
                    for uf in list(uf_to_muns.keys())[:2]:
                        self.debug_print(f"{uf}: {sorted(list(uf_to_muns[uf]))[:3]}...")
                    
                    self.debug_print("\nMunicipality -> Schools (first 2 municipalities):")
                    for mun in list(mun_to_schools.keys())[:2]:
                        self.debug_print(f"{mun}: {sorted(list(mun_to_schools[mun]))[:3]}...")
                    
                    self.debug_print(f"""
                        Hierarchical relationships built from CSV:
                        - {len(region_to_ufs)} regions
                        - {len(uf_to_muns)} UFs
                        - {len(mun_to_schools)} municipalities
                        - {sum(len(schools) for schools in mun_to_schools.values()):,} total schools
                    """)
                    
                    # Process schools in larger batches for better performance
                    batch_size = 1000
                    total_schools = sum(len(schools) for schools in mun_to_schools.values())
                    schools_processed = 0
                    
                    # Process municipalities in larger chunks
                    mun_batch_size = 50
                    total_muns = len(mun_to_schools)
                    muns_processed = 0
                    
                    self.debug_print(f"\nStarting update with batch sizes:")
                    self.debug_print(f"- Schools per batch: {batch_size:,}")
                    self.debug_print(f"- Municipalities per batch: {mun_batch_size:,}")
                    
                    # 1. Update UFs for each region
                    self.debug_print("\nUpdating UFs by region...")
                    for region, ufs in self.uf_by_region.items():
                        updates = [(region, uf) for uf in ufs]
                        cur.executemany("""
                            UPDATE dp_results_stats_bq
                            SET parent_regiao = %s,
                                parent_uf = NULL,
                                parent_municipio = NULL
                            WHERE group_by_col1 = 'SG_UF'
                            AND UPPER(group_by_val1) = UPPER(%s)
                            AND parent_regiao IS NULL;  -- Only update NULL values
                        """, updates)
                        self.debug_print(f"Updated {len(updates)} UFs for region {region}")
                    conn.commit()
                    
                    # 2. Update municipalities for each UF
                    self.debug_print("\nUpdating municipalities by UF...")
                    for uf, muns in self.mun_by_uf.items():
                        region = self.region_data[self.region_data['SG_UF'] == uf]['NO_REGIAO'].iloc[0]
                        updates = [(region, uf, mun) for mun in muns]
                        cur.executemany("""
                            UPDATE dp_results_stats_bq
                            SET parent_regiao = %s,
                                parent_uf = %s,
                                parent_municipio = NULL
                            WHERE group_by_col1 = 'NO_MUNICIPIO'
                            AND UPPER(group_by_val1) = UPPER(%s)
                            AND parent_regiao IS NULL;  -- Only update NULL values
                        """, updates)
                        
                        conn.commit()
                    
                    # 3. Update schools for each municipality
                    self.debug_print("\nUpdating schools...")
                    errors = []
                    schools_processed = 0
                    muns_processed = 0
                    
                    try:
                        for mun, schools in mun_to_schools.items():
                            try:
                                # Get UF and region from region_data using municipality name
                                mun_data = self.region_data[self.region_data['NO_MUNICIPIO'].str.upper() == mun]
                                if mun_data.empty:
                                    self.debug_print(f"Warning: Municipality not found: {mun}")
                                    errors.append(f"Municipality not found: {mun}")
                                    continue
                                
                                # Get the first matching UF and region
                                uf = mun_data['SG_UF'].iloc[0].upper()
                                region = mun_data['NO_REGIAO'].iloc[0].upper()
                                
                                # Process schools in batches
                                school_updates = []
                                for escola in schools:
                                    school_updates.append((region, uf, mun, escola))
                                    if len(school_updates) >= batch_size:
                                        try:
                                            cur.executemany("""
                                                UPDATE dp_results_stats_bq
                                                SET parent_regiao = %s,
                                                    parent_uf = %s,
                                                    parent_municipio = %s
                                                WHERE group_by_col1 = 'CO_ENTIDADE'
                                                AND group_by_val1 = %s
                                                AND parent_municipio IS NULL;
                                            """, school_updates)
                                            
                                            schools_processed += len(school_updates)
                                            if schools_processed % 10000 == 0:
                                                self.debug_print(f"""
                                                    Progress:
                                                    - Schools: {schools_processed:,}/{total_schools:,} ({schools_processed/total_schools*100:.1f}%)
                                                    - Municipalities: {muns_processed:,}/{total_muns:,} ({muns_processed/total_muns*100:.1f}%)
                                                    - Current municipality: {mun}
                                                    - Current UF: {uf}
                                                    - Current region: {region}
                                                """)
                                            
                                            school_updates = []
                                            conn.commit()
                                            
                                        except Exception as e:
                                            self.debug_print(f"Error in batch update: {str(e)}")
                                            errors.append(f"Batch error for {mun}: {str(e)}")
                                            conn.rollback()
                                            continue
                                
                                # Process remaining schools for this municipality
                                if school_updates:
                                    try:
                                        cur.executemany("""
                                            UPDATE dp_results_stats_bq
                                            SET parent_regiao = %s,
                                                parent_uf = %s,
                                                parent_municipio = %s
                                            WHERE group_by_col1 = 'CO_ENTIDADE'
                                            AND group_by_val1 = %s
                                            AND parent_municipio IS NULL;
                                        """, school_updates)
                                        schools_processed += len(school_updates)
                                        conn.commit()
                                    except Exception as e:
                                        self.debug_print(f"Error in final batch: {str(e)}")
                                        errors.append(f"Final batch error for {mun}: {str(e)}")
                                        conn.rollback()
                                        continue
                                
                                muns_processed += 1
                                
                            except Exception as e:
                                self.debug_print(f"Error processing municipality {mun}: {str(e)}")
                                errors.append(f"Municipality error: {mun}: {str(e)}")
                                continue
                        
                        # Final status report
                        self.debug_print(f"""
                            Update completed:
                            - Schools processed: {schools_processed:,}/{total_schools:,} ({schools_processed/total_schools*100:.1f}%)
                            - Municipalities processed: {muns_processed:,}/{total_muns:,} ({muns_processed/total_muns*100:.1f}%)
                            - Errors encountered: {len(errors)}
                        """)
                        
                        if errors:
                            self.debug_print("\nErrors encountered:")
                            for error in errors[:10]:  # Show first 10 errors
                                self.debug_print(error)
                            if len(errors) > 10:
                                self.debug_print(f"...and {len(errors) - 10} more errors")
                        
                    except Exception as e:
                        self.debug_print(f"Fatal error in school update: {str(e)}")
                        raise
            
        except Exception as e:
            self.debug_print(f"Error updating geographic hierarchy: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc()) 

