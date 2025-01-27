import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from google.cloud import bigquery
import os

# Configuração do banco PostgreSQL
PG_PARAMS = {
    'dbname': 'inep_synthetic_data',
    'user': 'postgres',
    'password': '0alxndr7',
    'host': 'localhost',
    'port': '5432'
}

# Configuração do BigQuery
project_id = 'nasnuvens-rpinep-rec-comp'
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

# Execute gcloud auth command with proper quoting for paths with spaces
os.system('"C:\\Users\\alexu\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.exe" auth application-default login')


class SyntheticDataVisualizationBQ:
    def __init__(self):
        """
        Inicializa a visualização dos dados sintéticos do BigQuery.
        """
        # Initialize BigQuery client
        self.project_id = 'nasnuvens-rpinep-rec-comp'

        # Execute gcloud auth command with proper quoting for paths with spaces
        os.system('"C:\\Users\\alexu\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.exe" auth application-default login')

        self.bq_client = bigquery.Client(project=self.project_id)

        # Create debug output with specific layout
        self.debug_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid black',
                padding='10px',
                height='200px',
                overflow='auto'
            )
        )
        
        # Create and connect widgets
        self._create_widgets()
        self._connect_observers()
        
    def _create_widgets(self):
        """
        Cria os widgets de controle para filtros.
        """
        # Geographic filters
        self.region_dropdown = widgets.Dropdown(
            options=['Todas'] + self._get_unique_values('no_regiao'),
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
        
        # Additional filters
        self.sex_dropdown = widgets.Dropdown(
            options=['Todos'] + self._get_unique_values('tp_sexo'),
            value='Todos',
            description='Sexo:'
        )
        
        self.age_dropdown = widgets.Dropdown(
            options=['Todos'] + self._get_unique_values('fx_etaria5'),
            value='Todos',
            description='Idade:'
        )
        
        self.race_dropdown = widgets.Dropdown(
            options=['Todos'] + self._get_unique_values('tp_raca'),
            value='Todos',
            description='Raça:'
        )
        
        self.school_dropdown = widgets.Dropdown(
            options=['Todos'],
            value='Todos',
            description='Escola:'
        )
        
        # Create figure widgets with specific layouts
        self.dist_fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                "Distribuição Geral",
                "Distribuição por Sexo (%)",
                "Distribuição por Raça (%)",
                "Distribuição por Faixa Etária (%)"
            ),
            figure=go.FigureWidget(
                layout=go.Layout(
                    height=1000,
                    width=1000,
                    showlegend=True,
                    template='plotly_white',
                    title_text="Análise da Distribuição da Nota de Redação"
                )
            )
        )
        
        # Update axes labels
        self.dist_fig.update_xaxes(title_text="Faixa de Notas")
        self.dist_fig.update_yaxes(title_text="Quantidade/Porcentagem de Estudantes")
        
        self.box_fig = go.FigureWidget(
            layout=go.Layout(
                height=800,
                width=1000,
                title_text="Análise da Nota de Redação por Características"
            )
        )
        
        self.stats_fig = go.FigureWidget(
            layout=go.Layout(
                height=800,
                width=1000,
                title_text="Análise Estatística da Nota de Redação"
            )
        )
        
        # Add a button to trigger data fetch and plot update
        self.update_button = widgets.Button(
            description='Atualizar Visualizações',
            button_style='primary'
        )
        
        # Add student distribution figure
        self.student_fig = go.FigureWidget(
            layout=go.Layout(
                height=800,
                width=1000,
                title_text="Distribuição dos Estudantes"
            )
        )
        
        self.debug_print("Widgets criados com sucesso")
        
    def _connect_observers(self):
        """
        Conecta os observadores aos widgets.
        """
        # Geographic cascading updates
        self.region_dropdown.observe(self._update_uf_options, names='value')
        self.uf_dropdown.observe(self._update_mun_options, names='value')
        self.mun_dropdown.observe(self._update_school_options, names='value')
        
        # Connect update button
        self.update_button.on_click(self._update_plots)
        
        self.debug_print("Observers connected")
        
    def _get_unique_values(self, column, condition_dict=None):
        """
        Obtém valores únicos de uma coluna com filtros opcionais do BigQuery.
        """
        try:
            if column == 'fx_etaria5':
                # Return predefined age groups matching gera_resultados_pd_bq.py
                return [
                    '1-5',
                    '6-10',
                    '11-15',
                    '16-20',
                    '21-plus',
                    'unknown'
                ]
            
            query = f"""
                SELECT DISTINCT {column}
                FROM `synthetic.ST_ENEM_2023`
                WHERE {column} IS NOT NULL
            """
            
            if condition_dict:
                conditions = []
                for key, value in condition_dict.items():
                    if value not in ['Todos', 'Todas']:
                        conditions.append(f"{key} = @{key}")
                
                if conditions:
                    query += " AND " + " AND ".join(conditions)
            
            query += f" ORDER BY {column}"
            
            job_config = bigquery.QueryJobConfig()
            if condition_dict:
                query_params = []
                for key, value in condition_dict.items():
                    if value not in ['Todos', 'Todas']:
                        query_params.append(bigquery.ScalarQueryParameter(key, "STRING", value))
                job_config.query_parameters = query_params
            
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            values = [row[0] for row in results if row[0] is not None]
            return values
            
        except Exception as e:
            self.debug_print(f"Error getting unique values for {column}: {str(e)}")
            return []

    def get_synthetic_data(self):
        """
        Obtém dados sintéticos com todos os filtros aplicados do BigQuery.
        """
        try:
            conditions = []
            query_params = []
            
            # Add all filter conditions with uppercase column names
            filter_values = {
                'NO_REGIAO': self.region_dropdown.value,
                'SG_UF': self.uf_dropdown.value,
                'NO_MUNICIPIO': self.mun_dropdown.value,
                'TP_SEXO': self.sex_dropdown.value,
                'TP_RACA': self.race_dropdown.value,
                'CO_ENTIDADE': self.school_dropdown.value
            }
            
            # Handle age filter separately since it needs to be calculated
            if self.age_dropdown.value != 'Todos':
                age_range = self.age_dropdown.value
                if age_range == '21-plus':
                    conditions.append("NU_IDADE >= 21")
                else:
                    min_age, max_age = map(int, age_range.split('-'))
                    conditions.append(f"NU_IDADE BETWEEN @min_age AND @max_age")
                    query_params.extend([
                        bigquery.ScalarQueryParameter("min_age", "INT64", min_age),
                        bigquery.ScalarQueryParameter("max_age", "INT64", max_age)
                    ])
            
            for column, value in filter_values.items():
                if value not in ['Todos', 'Todas']:
                    conditions.append(f"{column} = @{column}")
                    query_params.append(bigquery.ScalarQueryParameter(column, "STRING", value))
            
            query = """
                SELECT 
                    CO_PESSOA_FISICA,
                    NO_REGIAO,
                    SG_UF,
                    NO_MUNICIPIO,
                    CO_ENTIDADE,
                    TP_SEXO,
                    TP_RACA,
                    NU_IDADE,
                    NU_NOTA_REDACAO,
                    CASE 
                        WHEN NU_IDADE BETWEEN 1 AND 5 THEN '1-5'
                        WHEN NU_IDADE BETWEEN 6 AND 10 THEN '6-10'
                        WHEN NU_IDADE BETWEEN 11 AND 15 THEN '11-15'
                        WHEN NU_IDADE BETWEEN 16 AND 20 THEN '16-20'
                        WHEN NU_IDADE >= 21 THEN '21-plus'
                        ELSE 'unknown'
                    END as FX_ETARIA5
                FROM `synthetic.ST_ENEM_2023`
            """
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Add LIMIT to prevent memory issues
            query += " LIMIT 100000"
            
            self.debug_print(f"Executing query: {query}")
            if query_params:
                self.debug_print(f"With parameters: {query_params}")
            
            job_config = bigquery.QueryJobConfig()
            job_config.query_parameters = query_params
            
            query_job = self.bq_client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            # Convert column names to lowercase to match existing code
            df.columns = df.columns.str.lower()
            
            return df
            
        except Exception as e:
            self.debug_print(f"Error getting synthetic data: {str(e)}")
            return pd.DataFrame()

    def _update_uf_options(self, change):
        """
        Atualiza as opções de UF baseado na região selecionada.
        """
        region = change['new']
        try:
            if region == 'Todas':
                query = """
                    SELECT DISTINCT SG_UF
                    FROM `synthetic.ST_ENEM_2023`
                    WHERE SG_UF IS NOT NULL
                    ORDER BY SG_UF
                """
                job_config = bigquery.QueryJobConfig()
            else:
                query = """
                    SELECT DISTINCT SG_UF
                    FROM `synthetic.ST_ENEM_2023`
                    WHERE NO_REGIAO = @region
                    AND SG_UF IS NOT NULL
                    ORDER BY SG_UF
                """
                job_config = bigquery.QueryJobConfig()
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter("region", "STRING", region)
                ]
            
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            ufs = ['Todas'] + [row.SG_UF for row in results]
            self.uf_dropdown.options = ufs
            self.uf_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error updating UFs: {str(e)}")
            
    def _update_mun_options(self, change):
        """
        Atualiza as opções de municípios baseado na UF selecionada.
        """
        uf = change['new']
        try:
            if uf == 'Todas':
                if self.region_dropdown.value == 'Todas':
                    query = """
                        SELECT DISTINCT NO_MUNICIPIO
                        FROM `synthetic.ST_ENEM_2023`
                        WHERE NO_MUNICIPIO IS NOT NULL
                        ORDER BY NO_MUNICIPIO
                    """
                    job_config = bigquery.QueryJobConfig()
                else:
                    query = """
                        SELECT DISTINCT NO_MUNICIPIO
                        FROM `synthetic.ST_ENEM_2023`
                        WHERE NO_REGIAO = @region
                        AND NO_MUNICIPIO IS NOT NULL
                        ORDER BY NO_MUNICIPIO
                    """
                    job_config = bigquery.QueryJobConfig()
                    job_config.query_parameters = [
                        bigquery.ScalarQueryParameter("region", "STRING", self.region_dropdown.value)
                    ]
            else:
                query = """
                    SELECT DISTINCT NO_MUNICIPIO
                    FROM `synthetic.ST_ENEM_2023`
                    WHERE SG_UF = @uf
                    AND NO_MUNICIPIO IS NOT NULL
                    ORDER BY NO_MUNICIPIO
                """
                job_config = bigquery.QueryJobConfig()
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter("uf", "STRING", uf)
                ]
            
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            municipalities = ['Todas'] + [row.NO_MUNICIPIO for row in results]
            self.mun_dropdown.options = municipalities
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error updating municipalities: {str(e)}")
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
    
    def _update_school_options(self, change):
        """
        Atualiza as opções de escolas baseado nos filtros geográficos.
        """
        try:
            conditions = {}
            
            if self.region_dropdown.value != 'Todas':
                conditions['no_regiao'] = self.region_dropdown.value
                
            if self.uf_dropdown.value != 'Todas':
                conditions['sg_uf'] = self.uf_dropdown.value
                
            if self.mun_dropdown.value != 'Todas':
                conditions['no_municipio'] = self.mun_dropdown.value
                
            schools = ['Todos'] + self._get_unique_values('co_entidade', conditions)
            self.school_dropdown.options = schools
            self.school_dropdown.value = 'Todos'
            
        except Exception as e:
            self.debug_print(f"Error updating schools: {str(e)}")
    
    def debug_print(self, message):
        """
        Imprime mensagens de debug no widget de output.
        """
        with self.debug_output:
            print(message)
            
    def display(self):
        """
        Exibe a interface de visualização.
        """
        try:
            # Create filter containers
            geo_filters = widgets.HBox([
                self.region_dropdown,
                self.uf_dropdown,
                self.mun_dropdown
            ])
            
            other_filters = widgets.HBox([
                self.sex_dropdown,
                self.age_dropdown,
                self.race_dropdown,
                self.school_dropdown
            ])
            '''
            # Create main container
            self.container = widgets.VBox([
                geo_filters,
                other_filters,
                self.update_button,
                self.dist_fig,
                self.box_fig,
                self.stats_fig,
                self.student_fig,
                self.debug_output
            ], layout=widgets.Layout(height='3200px', width='100%'))
            '''
            # Create main container
            self.container = widgets.VBox([
                geo_filters,
                other_filters,
                self.update_button,
                self.dist_fig,
                self.box_fig,
                self.stats_fig,
                self.student_fig,
                self.debug_output
            ])            
            self.debug_print("Interface initialized")
            display(self.container)
            
        except Exception as e:
            self.debug_print(f"Error in display: {str(e)}")

    def _update_plots(self, _):
        """
        Atualiza todos os gráficos.
        """
        try:
            self.debug_print("\n=== Iniciando atualização dos plots ===")
            
            # Get filtered data
            data = self.get_synthetic_data()
            
            if data.empty:
                self.debug_print("No data available for selected filters")
                return
                
            # Update all plots
            self._update_distribution_plot(data)
            self._update_boxplots(data)
            self._update_stats_plot(data)
            self._update_student_distribution(data)
            
            self.debug_print("\n=== Plots atualizados com sucesso ===")
            
        except Exception as e:
            self.debug_print(f"Error updating plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_distribution_plot(self, data):
        """
        Atualiza o gráfico de distribuição com faixas de notas e análise por subgrupos.
        """
        try:
            self.debug_print("\n=== Atualizando o gráfico de distribuição ===")
            
            # Create grade brackets (0-100 range)
            data['faixa_nota'] = pd.cut(
                data['nu_nota_redacao'],
                bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                labels=['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
            )
            
            with self.dist_fig.batch_update():
                # Clear existing traces
                self.dist_fig.data = []
                
                # Overall distribution with grade brackets
                grade_dist = data['faixa_nota'].value_counts().sort_index()
                self.dist_fig.add_trace(
                    go.Bar(
                        x=grade_dist.index,
                        y=grade_dist.values,
                        name='Distribuição Geral',
                        text=grade_dist.values,
                        textposition='outside',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Distribution by gender
                for sexo in sorted(data['tp_sexo'].unique()):
                    sexo_data = data[data['tp_sexo'] == sexo]
                    sexo_dist = sexo_data['faixa_nota'].value_counts().sort_index()
                    pct_sexo = (sexo_dist / sexo_dist.sum()) * 100
                    
                    self.debug_print(f"\nGender {sexo} distribution:\n{pct_sexo}")
                    
                    self.dist_fig.add_trace(
                        go.Scatter(
                            x=pct_sexo.index,
                            y=pct_sexo.values,
                            name=f'Sexo: {sexo}',
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=8)
                        ),
                        row=1, col=2
                    )
                
                # Distribution by race
                for raca in sorted(data['tp_raca'].unique()):
                    raca_data = data[data['tp_raca'] == raca]
                    raca_dist = raca_data['faixa_nota'].value_counts().sort_index()
                    pct_raca = (raca_dist / raca_dist.sum()) * 100
                    
                    self.debug_print(f"\nRace {raca} distribution:\n{pct_raca}")
                    
                    self.dist_fig.add_trace(
                        go.Scatter(
                            x=pct_raca.index,
                            y=pct_raca.values,
                            name=f'Raça: {raca}',
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=8)
                        ),
                        row=2, col=1
                    )
                
                # Age group distribution
                for faixa in sorted(data['fx_etaria5'].unique()):
                    idade_data = data[data['fx_etaria5'] == faixa]
                    idade_dist = idade_data['faixa_nota'].value_counts().sort_index()
                    pct_idade = (idade_dist / idade_dist.sum()) * 100
                    
                    self.debug_print(f"\nAge group {faixa} distribution:\n{pct_idade}")
                    
                    self.dist_fig.add_trace(
                        go.Scatter(
                            x=pct_idade.index,
                            y=pct_idade.values,
                            name=f'Idade: {faixa}',
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=8)
                        ),
                        row=2, col=2
                    )
                
                # Update layout
                self.dist_fig.update_layout(
                    showlegend=True,
                    template='plotly_white',
                    height=1000,
                    title_text="Análise da Distribuição da Nota de Redação",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
            
            #self.debug_print(f"\nPlot updated with {len(self.dist_fig.data)} traces")
            
        except Exception as e:
            self.debug_print(f"Error updating distribution plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_boxplots(self, data):
        """
        Atualiza os box plots.
        """
        try:
            # Clear existing traces
            self.box_fig.data = []
            
            # Add new traces
            self.box_fig.add_trace(
                go.Box(
                    y=data['nu_nota_redacao'],
                    x=data['tp_sexo'],
                    name='Por Sexo',
                    boxpoints='outliers',
                    notched=True
                )
            )
            
            self.box_fig.add_trace(
                go.Box(
                    y=data['nu_nota_redacao'],
                    x=data['tp_raca'],
                    name='Por Raça',
                    boxpoints='outliers',
                    notched=True
                )
            )
            
            # Update layout with better labels
            self.box_fig.update_layout(
                title_text="Análise da Nota de Redação por Características",
                showlegend=True,
                template='plotly_white',
                xaxis_title="Categoria",
                yaxis_title="Nota da Redação",
                height=800
            )
            
            self.debug_print("Box plots updated with new traces")
            
        except Exception as e:
            self.debug_print(f"Error updating box plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_stats_plot(self, data):
        """
        Atualiza o gráfico de estatísticas.
        """
        try:
            # Clear existing traces
            self.stats_fig.data = []
            
            # Calculate statistics
            stats = {
                'Média': data['nu_nota_redacao'].mean(),
                'Mediana': data['nu_nota_redacao'].median(),
                'Desvio Padrão': data['nu_nota_redacao'].std(),
                'IQR': data['nu_nota_redacao'].quantile(0.75) - data['nu_nota_redacao'].quantile(0.25)
            }
            
            # Add statistics trace
            self.stats_fig.add_trace(
                go.Bar(
                    x=list(stats.keys()),
                    y=list(stats.values()),
                    name='Nota',
                    text=[f'{v:.2f}' for v in stats.values()],
                    textposition='outside'
                )
            )
            
            # Update layout with better labels
            self.stats_fig.update_layout(
                title_text="Análise Estatística da Nota de Redação",
                showlegend=True,
                template='plotly_white',
                xaxis_title="Medida Estatística",
                yaxis_title="Valor",
                height=800
            )
            
            self.debug_print("Stats plot updated with new traces")
            
        except Exception as e:
            self.debug_print(f"Error updating stats plot: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_student_distribution(self, data):
        """
        Cria visualização da distribuição de estudantes por diferentes categorias.
        """
        try:
            # Calculate distributions first to check data
            sex_dist = data['tp_sexo'].value_counts()
            race_dist = data['tp_raca'].value_counts()
            age_dist = data['fx_etaria5'].value_counts()
            region_dist = data['no_regiao'].value_counts()
            
            self.debug_print(f"Sex distribution: {len(sex_dist)} categories")
            self.debug_print(f"Race distribution: {len(race_dist)} categories")
            self.debug_print(f"Age distribution: {len(age_dist)} categories")
            self.debug_print(f"Region distribution: {len(region_dist)} categories")
            
            # Clear existing traces
            self.student_fig.data = []
            
            # Update layout first
            self.student_fig.update_layout(
                title_text="Distribuição dos Estudantes",
                showlegend=True,
                template='plotly_white',
                height=800,
                width=1000,
                grid={'rows': 2, 'columns': 2},
                annotations=[
                    dict(text="Distribuição por Sexo", x=0.12, y=1.05, showarrow=False, xref='paper', yref='paper'),
                    dict(text="Distribuição por Raça", x=0.85, y=1.05, showarrow=False, xref='paper', yref='paper'),
                    dict(text="Distribuição por Faixa Etária", x=0.12, y=0.50, showarrow=False, xref='paper', yref='paper'),
                    dict(text="Distribuição por Região", x=0.85, y=0.50, showarrow=False, xref='paper', yref='paper')
                ]
            )
            
            # Add pie charts
            self.student_fig.add_trace(
                go.Pie(
                    labels=sex_dist.index,
                    values=sex_dist.values,
                    name='Sexo',
                    textinfo='percent+label',
                    hole=0.3,
                    domain={'x': [0, 0.45], 'y': [0.55, 1]}
                )
            )
            
            self.student_fig.add_trace(
                go.Pie(
                    labels=race_dist.index,
                    values=race_dist.values,
                    name='Raça',
                    textinfo='percent+label',
                    hole=0.3,
                    domain={'x': [0.55, 1], 'y': [0.55, 1]}
                )
            )
            
            # Add bar charts
            self.student_fig.add_trace(
                go.Bar(
                    x=age_dist.index,
                    y=age_dist.values,
                    name='Faixa Etária',
                    text=age_dist.values,
                    textposition='outside',
                    xaxis='x1',
                    yaxis='y1'
                )
            )
            
            self.student_fig.add_trace(
                go.Bar(
                    x=region_dist.index,
                    y=region_dist.values,
                    name='Região',
                    text=region_dist.values,
                    textposition='outside',
                    xaxis='x2',
                    yaxis='y2'
                )
            )
            
            # Update axes for bar charts
            self.student_fig.update_layout(
                xaxis1=dict(
                    domain=[0, 0.45],
                    anchor='y1',
                    title='Faixa Etária'
                ),
                xaxis2=dict(
                    domain=[0.55, 1],
                    anchor='y2',
                    title='Região'
                ),
                yaxis1=dict(
                    domain=[0, 0.45],
                    anchor='x1',
                    title='Quantidade de Estudantes'
                ),
                yaxis2=dict(
                    domain=[0, 0.45],
                    anchor='x2',
                    title='Quantidade de Estudantes'
                )
            )
            
            self.debug_print("Student distribution plot updated with:")
            self.debug_print(f"- {len(self.student_fig.data)} traces")
            self.debug_print(f"- Layout: {self.student_fig.layout.width}x{self.student_fig.layout.height}")
            
        except Exception as e:
            self.debug_print(f"Error updating student distribution: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc()) 