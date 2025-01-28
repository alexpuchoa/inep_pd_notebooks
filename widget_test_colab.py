import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import numpy as np

class WidgetTest:
    """Test different widget and display combinations in Colab."""
    
    def __init__(self):
        """Initialize with different widget combinations."""
        print("Testing different widget combinations:")
        
        # Test 1: Simple dropdown and button
        print("\nTest 1: Simple widgets with individual display")
        self.dropdown1 = widgets.Dropdown(
            options=['Option 1', 'Option 2', 'Option 3'],
            description='Test 1:'
        )
        self.button1 = widgets.Button(description='Update 1')
        
        display(widgets.HTML("<h3>Test 1: Individual Widgets</h3>"))
        display(self.dropdown1)
        display(self.button1)
        
        # Test 2: Widgets in HBox
        print("\nTest 2: Widgets in HBox")
        self.dropdown2 = widgets.Dropdown(
            options=['A', 'B', 'C'],
            description='Test 2:'
        )
        self.button2 = widgets.Button(description='Update 2')
        
        display(widgets.HTML("<h3>Test 2: HBox Container</h3>"))
        display(widgets.HBox([self.dropdown2, self.button2]))
        
        # Test 3: Widgets in VBox with Output
        print("\nTest 3: VBox with Output widget")
        self.dropdown3 = widgets.Dropdown(
            options=['X', 'Y', 'Z'],
            description='Test 3:'
        )
        self.button3 = widgets.Button(description='Update 3')
        self.output3 = widgets.Output()
        
        display(widgets.HTML("<h3>Test 3: VBox with Output</h3>"))
        display(widgets.VBox([
            self.dropdown3,
            self.button3,
            self.output3
        ]))
        
        # Test 4: Plotly figure with Output
        print("\nTest 4: Plotly figure in Output widget")
        self.button4 = widgets.Button(description='Update Plot')
        self.plot_output = widgets.Output()
        
        display(widgets.HTML("<h3>Test 4: Plotly Plot</h3>"))
        display(self.button4)
        display(self.plot_output)
        
        # Connect observers
        self.button1.on_click(self.update_test1)
        self.button2.on_click(self.update_test2)
        self.button3.on_click(self.update_test3)
        self.button4.on_click(self.update_plot)
        
        # Initial plot
        self.update_plot(None)
    
    def update_test1(self, b):
        print(f"Test 1 selected: {self.dropdown1.value}")
    
    def update_test2(self, b):
        print(f"Test 2 selected: {self.dropdown2.value}")
    
    def update_test3(self, b):
        with self.output3:
            clear_output()
            print(f"Test 3 selected: {self.dropdown3.value}")
    
    def update_plot(self, b):
        with self.plot_output:
            clear_output(wait=True)
            
            # Create sample data
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
            fig.update_layout(
                title='Sample Plot',
                height=400,
                width=600
            )
            
            display(fig)

# Create and display test app
test_app = WidgetTest() 