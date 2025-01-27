import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import psycopg2
import psycopg2.extras

# Configuração do banco PostgreSQL
PG_PARAMS = {
    'dbname': 'inep_synthetic_data',
    'user': 'postgres',
    'password': '0alxndr7',
    'host': 'localhost',
    'port': '5432'
}

class SyntheticDataVisualization:
    def __init__(self, conn_params):
        """
        Inicializa a visualização dos dados sintéticos.
        """
        self.conn_params = conn_params
        
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
        try:
            # Connect region change to UF update
            self.region_dropdown.observe(self._update_uf_options, names='value')
            
            # Connect UF change to municipality update
            self.uf_dropdown.observe(self._update_mun_options, names='value')
            
            # Connect update button to plot updates
            self.update_button.on_click(self._update_plots)
            
            self.debug_print("Observers connected")
            
        except Exception as e:
            self.debug_print(f"Error connecting observers: {str(e)}")
    
    def debug_print(self, message):
        """
        Imprime mensagem de debug na área de output.
        """
        with self.debug_output:
            print(message)
    
    def _get_unique_values(self, column, condition_dict=None):
        """
        Obtém valores únicos de uma coluna com filtros opcionais.
        """
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            
            query = f"SELECT DISTINCT {column} FROM synthetic_data WHERE {column} IS NOT NULL"
            
            if condition_dict:
                conditions = []
                params = []
                for key, value in condition_dict.items():
                    if value not in ['Todos', 'Todas']:
                        conditions.append(f"{key} = %s")
                        params.append(value)
                
                if conditions:
                    query += " AND " + " AND ".join(conditions)
                    cur.execute(query + " ORDER BY " + column, tuple(params))
                else:
                    cur.execute(query + " ORDER BY " + column)
            else:
                cur.execute(query + " ORDER BY " + column)
            
            values = [row[0] for row in cur.fetchall()]
            return values
            
        except Exception as e:
            self.debug_print(f"Error getting unique values for {column}: {str(e)}")
            return []
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close() 

    def _update_uf_options(self, change):
        """
        Atualiza as opções de UF baseado na região selecionada.
        """
        region = change['new']
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            
            if region == 'Todas':
                cur.execute("""
                    SELECT DISTINCT sg_uf 
                    FROM synthetic_data 
                    WHERE sg_uf IS NOT NULL 
                    ORDER BY sg_uf
                """)
            else:
                cur.execute("""
                    SELECT DISTINCT sg_uf 
                    FROM synthetic_data 
                    WHERE no_regiao = %s 
                    AND sg_uf IS NOT NULL 
                    ORDER BY sg_uf
                """, (region,))
            
            ufs = ['Todas'] + [row[0] for row in cur.fetchall()]
            self.uf_dropdown.options = ufs
            self.uf_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error updating UFs: {str(e)}")
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()
    
    def _update_mun_options(self, change):
        """
        Atualiza as opções de municípios baseado na UF selecionada.
        """
        uf = change['new']
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            
            if uf == 'Todas':
                if self.region_dropdown.value == 'Todas':
                    cur.execute("""
                        SELECT DISTINCT no_municipio 
                        FROM synthetic_data 
                        WHERE no_municipio IS NOT NULL 
                        ORDER BY no_municipio
                    """)
                else:
                    cur.execute("""
                        SELECT DISTINCT no_municipio 
                        FROM synthetic_data 
                        WHERE no_regiao = %s 
                        AND no_municipio IS NOT NULL 
                        ORDER BY no_municipio
                    """, (self.region_dropdown.value,))
            else:
                cur.execute("""
                    SELECT DISTINCT no_municipio 
                    FROM synthetic_data 
                    WHERE sg_uf = %s 
                    AND no_municipio IS NOT NULL 
                    ORDER BY no_municipio
                """, (uf,))
            
            municipalities = ['Todas'] + [row[0] for row in cur.fetchall()]
            self.mun_dropdown.options = municipalities
            self.mun_dropdown.value = 'Todas'
            
        except Exception as e:
            self.debug_print(f"Error updating municipalities: {str(e)}")
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close() 

    def get_synthetic_data(self):
        """
        Obtém dados sintéticos com todos os filtros aplicados.
        """
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            
            conditions = []
            params = []
            
            # Add all filter conditions
            filter_values = {
                'no_regiao': self.region_dropdown.value,
                'sg_uf': self.uf_dropdown.value,
                'no_municipio': self.mun_dropdown.value,
                'tp_sexo': self.sex_dropdown.value,
                'fx_etaria5': self.age_dropdown.value,
                'tp_raca': self.race_dropdown.value,
                'co_entidade': self.school_dropdown.value
            }
            
            for column, value in filter_values.items():
                if value not in ['Todos', 'Todas']:
                    conditions.append(f"{column} = %s")
                    params.append(value)
            
            query = """
                SELECT 
                    co_pessoa_fisica,
                    no_regiao,
                    sg_uf,
                    no_municipio,
                    co_entidade,
                    tp_sexo,
                    tp_raca,
                    nu_idade,
                    nu_nota_redacao,
                    fx_etaria5
                FROM synthetic_data
            """
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Add LIMIT to prevent memory issues
            query += " LIMIT 100000"
            
            self.debug_print(f"Executing query: {query}")
            if params:
                self.debug_print(f"With parameters: {params}")
                
            cur.execute(query, tuple(params))
            
            columns = [desc[0] for desc in cur.description]
            data = pd.DataFrame(cur.fetchall(), columns=columns)
            
            return data
            
        except Exception as e:
            self.debug_print(f"Error getting synthetic data: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close() 

    def _update_plots(self, _):
        """
        Atualiza todos os gráficos com os dados filtrados.
        """
        try:
            # Get filtered data
            df = self.get_synthetic_data()
            
            if df.empty:
                self.debug_print("No data available for the selected filters")
                return
            
            # Update distribution plots
            self._update_distribution_plots(df)
            
            # Update box plots
            self._update_box_plots(df)
            
            # Update statistics plots
            self._update_statistics_plots(df)
            
            # Update student distribution
            self._update_student_distribution(df)
            
            self.debug_print("All plots updated successfully")
            
        except Exception as e:
            self.debug_print(f"Error updating plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())
    
    def _update_distribution_plots(self, df):
        """
        Atualiza os gráficos de distribuição.
        """
        try:
            # Clear existing traces
            self.dist_fig.data = []
            
            # General distribution
            self.dist_fig.add_trace(
                go.Histogram(
                    x=df['nu_nota_redacao'],
                    name='Geral',
                    nbinsx=20
                ),
                row=1, col=1
            )
            
            # Distribution by sex
            sex_dist = df.groupby('tp_sexo')['nu_nota_redacao'].value_counts(normalize=True).unstack()
            for sex in sex_dist.index:
                self.dist_fig.add_trace(
                    go.Bar(
                        x=sex_dist.columns,
                        y=sex_dist.loc[sex] * 100,
                        name=sex
                    ),
                    row=1, col=2
                )
            
            # Distribution by race
            race_dist = df.groupby('tp_raca')['nu_nota_redacao'].value_counts(normalize=True).unstack()
            for race in race_dist.index:
                self.dist_fig.add_trace(
                    go.Bar(
                        x=race_dist.columns,
                        y=race_dist.loc[race] * 100,
                        name=race
                    ),
                    row=2, col=1
                )
            
            # Distribution by age group
            age_dist = df.groupby('fx_etaria5')['nu_nota_redacao'].value_counts(normalize=True).unstack()
            for age in age_dist.index:
                self.dist_fig.add_trace(
                    go.Bar(
                        x=age_dist.columns,
                        y=age_dist.loc[age] * 100,
                        name=age
                    ),
                    row=2, col=2
                )
            
            self.debug_print("Distribution plots updated with:")
            self.debug_print(f"- {len(df)} total records")
            self.debug_print(f"- {len(self.dist_fig.data)} traces")
            
        except Exception as e:
            self.debug_print(f"Error updating distribution plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_box_plots(self, df):
        """
        Atualiza os box plots.
        """
        try:
            # Clear existing traces
            self.box_fig.data = []
            
            # Box plot by sex
            sex_data = [
                go.Box(
                    y=df[df['tp_sexo'] == sex]['nu_nota_redacao'],
                    name=sex,
                    boxpoints='outliers'
                )
                for sex in df['tp_sexo'].unique()
            ]
            
            # Box plot by race
            race_data = [
                go.Box(
                    y=df[df['tp_raca'] == race]['nu_nota_redacao'],
                    name=race,
                    boxpoints='outliers'
                )
                for race in df['tp_raca'].unique()
            ]
            
            # Box plot by age group
            age_data = [
                go.Box(
                    y=df[df['fx_etaria5'] == age]['nu_nota_redacao'],
                    name=age,
                    boxpoints='outliers'
                )
                for age in sorted(df['fx_etaria5'].unique())
            ]
            
            # Add all traces
            for trace in sex_data + race_data + age_data:
                self.box_fig.add_trace(trace)
            
            # Update layout
            self.box_fig.update_layout(
                xaxis_title="Características",
                yaxis_title="Nota de Redação",
                boxmode='group'
            )
            
            self.debug_print("Box plots updated successfully")
            
        except Exception as e:
            self.debug_print(f"Error updating box plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())
    
    def _update_statistics_plots(self, df):
        """
        Atualiza os gráficos de estatísticas.
        """
        try:
            # Clear existing traces
            self.stats_fig.data = []
            
            # Calculate statistics by different groups
            stats_by_sex = df.groupby('tp_sexo')['nu_nota_redacao'].agg(['mean', 'median', 'std']).round(2)
            stats_by_race = df.groupby('tp_raca')['nu_nota_redacao'].agg(['mean', 'median', 'std']).round(2)
            stats_by_age = df.groupby('fx_etaria5')['nu_nota_redacao'].agg(['mean', 'median', 'std']).round(2)
            
            # Add traces for each statistic
            categories = ['Por Sexo', 'Por Raça', 'Por Idade']
            stats_dfs = [stats_by_sex, stats_by_race, stats_by_age]
            
            for i, (category, stats_df) in enumerate(zip(categories, stats_dfs)):
                # Mean
                self.stats_fig.add_trace(
                    go.Bar(
                        x=[f"{category} - {idx}" for idx in stats_df.index],
                        y=stats_df['mean'],
                        name=f'Média ({category})',
                        error_y=dict(
                            type='data',
                            array=stats_df['std'],
                            visible=True
                        )
                    )
                )
                
                # Median
                self.stats_fig.add_trace(
                    go.Scatter(
                        x=[f"{category} - {idx}" for idx in stats_df.index],
                        y=stats_df['median'],
                        name=f'Mediana ({category})',
                        mode='markers',
                        marker=dict(size=10, symbol='diamond')
                    )
                )
            
            # Update layout
            self.stats_fig.update_layout(
                xaxis_title="Grupos",
                yaxis_title="Nota de Redação",
                barmode='group'
            )
            
            self.debug_print("Statistics plots updated successfully")
            
        except Exception as e:
            self.debug_print(f"Error updating statistics plots: {str(e)}")
            import traceback
            self.debug_print(traceback.format_exc())

    def _update_student_distribution(self, df):
        """
        Atualiza o gráfico de distribuição dos estudantes.
        """
        try:
            # Clear existing traces
            self.student_fig.data = []
            
            # Calculate distributions
            age_dist = df['fx_etaria5'].value_counts()
            region_dist = df['no_regiao'].value_counts()
            
            # Update layout for subplots
            self.student_fig.update_layout(
                grid=dict(
                    rows=2,
                    columns=2,
                    pattern='independent'
                ),
                title_text="Distribuição dos Estudantes",
                showlegend=False
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

    def display(self):
        """
        Organiza e exibe todos os widgets e gráficos.
        """
        # Create filter controls layout
        filters = widgets.VBox([
            widgets.HBox([
                self.region_dropdown,
                self.uf_dropdown,
                self.mun_dropdown
            ]),
            widgets.HBox([
                self.sex_dropdown,
                self.age_dropdown,
                self.race_dropdown,
                self.school_dropdown
            ]),
            self.update_button
        ])
        
        # Create plots layout
        plots = widgets.VBox([
            self.dist_fig,
            self.box_fig,
            self.stats_fig,
            self.student_fig
        ])
        
        # Display debug output, filters and plots
        display(
            widgets.VBox([
                self.debug_output,
                filters,
                plots
            ])
        )