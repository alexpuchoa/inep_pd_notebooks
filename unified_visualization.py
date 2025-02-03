import pandas as pd
from pathlib import Path
from visual_notebook_v2 import VisualizationNotebook
from visual_notebook_zeradas_table import TableVisualization
from visual_notebook_zeradas_colab import LostValuesVisualizationColab

class UnifiedVisualization:
    """Class that combines all three visualization types"""
    
    def __init__(self, data_path):
        """
        Initialize with data path and load data once
        """
        try:
            print("Initializing Unified Visualization...")
            self.data_path = Path(data_path)
            
            # Load data once
            print("Loading data...")
            csv_path = self.data_path / "dp_results_stats_bq.csv"
            queries_file = self.data_path / "queries_formatadas_bq.csv"
            
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
            
            self.data = pd.concat(chunks, ignore_index=True)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Load queries configuration
            self.queries_config = pd.read_csv(queries_file, sep=';')
            
            # Create visualization instances but don't display them yet
            self.viz_v2 = None
            self.viz_table = None
            self.viz_zeradas = None
            
            print("Unified Visualization ready!")
            
        except Exception as e:
            print(f"Error in UnifiedVisualization initialization: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def show_v2(self):
        """Display the main visualization"""
        if self.viz_v2 is None:
            self.viz_v2 = VisualizationNotebook(data=self.data, queries_config=self.queries_config, data_path=self.data_path)
    
    def show_table(self):
        """Display the table visualization"""
        if self.viz_table is None:
            self.viz_table = TableVisualization(data=self.data, queries_config=self.queries_config)
    
    def show_zeradas(self):
        """Display the lost values visualization"""
        if self.viz_zeradas is None:
            self.viz_zeradas = LostValuesVisualizationColab(data=self.data, queries_config=self.queries_config) 