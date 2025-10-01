import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Gate Classification Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .gate-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PerceptronGate:
    """Single Layer Perceptron for Gate Classification"""
    
    def __init__(self, gate_type):
        self.gate_type = gate_type
        self.weights = None
        self.bias = None
        self.training_history = []
        
    def step_function(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        """Make predictions"""
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained yet!")
        linear_output = np.dot(X, self.weights) + self.bias
        return self.step_function(linear_output)
    
    def train(self, X, y, learning_rate=0.1, epochs=20, init_method='random'):
        """Train the perceptron"""
        self.training_history = []
        
        # Initialize weights and bias
        if init_method == 'random':
            np.random.seed(42)  # For reproducibility
            self.weights = np.random.randn(X.shape[1])
            self.bias = np.random.randn(1)
        else:  # zeros
            self.weights = np.zeros(X.shape[1])
            self.bias = np.zeros(1)
        
        # Training loop
        for epoch in range(epochs):
            epoch_errors = []
            for xi, target in zip(X, y):
                # Forward pass
                linear_output = np.dot(xi, self.weights) + self.bias
                pred = self.step_function(linear_output)
                error = target - pred
                
                # Update weights and bias
                self.weights += learning_rate * error * xi
                self.bias += learning_rate * error
                
                epoch_errors.append(abs(error))
            
            # Store training history
            avg_error = np.mean(epoch_errors)
            self.training_history.append({
                'epoch': epoch + 1,
                'avg_error': avg_error,
                'weights': self.weights.copy(),
                'bias': self.bias.copy()
            })
    
    def get_decision_boundary_data(self, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5), resolution=100):
        """Generate decision boundary data for plotting"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        return xx, yy, Z

def get_gate_data(gate_type):
    """Get truth table data for different gates"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    if gate_type == "AND":
        y = np.array([0, 0, 0, 1])
        description = "AND gate outputs 1 only if both inputs are 1"
    elif gate_type == "OR":
        y = np.array([0, 1, 1, 1])
        description = "OR gate outputs 1 if at least one input is 1"
    elif gate_type == "AND-NOT":
        y = np.array([0, 0, 1, 0])
        description = "AND-NOT gate outputs 1 only if first input is 1 and second input is 0"
    elif gate_type == "XOR":
        y = np.array([0, 1, 1, 0])
        description = "XOR gate outputs 1 only when inputs are different"
    
    return X, y, description

def create_truth_table_display(X, y, gate_type):
    """Create a nicely formatted truth table"""
    df = pd.DataFrame({
        'Input 1': X[:, 0],
        'Input 2': X[:, 1],
        f'{gate_type} Output': y
    })
    return df

def plot_decision_boundary(perceptron, X, y, gate_type):
    """Create interactive decision boundary plot"""
    xx, yy, Z = perceptron.get_decision_boundary_data()
    
    # Create the plot
    fig = go.Figure()
    
    # Add decision boundary contour
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale=[[0, 'lightcoral'], [1, 'lightblue']],
        opacity=0.6,
        showscale=False,
        contours=dict(start=0, end=1, size=1),
        name="Decision Boundary"
    ))
    
    # Add data points
    colors = ['red' if label == 0 else 'blue' for label in y]
    symbols = ['circle' if label == 0 else 'x' for label in y]
    
    for i, (point, label, color, symbol) in enumerate(zip(X, y, colors, symbols)):
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]],
            mode='markers',
            marker=dict(
                size=15,
                color=color,
                symbol=symbol,
                line=dict(width=2, color='black')
            ),
            name=f"Class {label}" if i == 0 or (i == 1 and y[0] != y[1]) else "",
            showlegend=bool(i == 0 or (i == 1 and y[0] != y[1])),
            text=f"Input: ({point[0]}, {point[1]})<br>Output: {label}",
            hovertemplate="%{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"{gate_type} Gate Decision Boundary",
        xaxis_title="Input 1",
        yaxis_title="Input 2",
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

def plot_training_progress(perceptron):
    """Plot training progress"""
    if not perceptron.training_history:
        return None
    
    epochs = [h['epoch'] for h in perceptron.training_history]
    errors = [h['avg_error'] for h in perceptron.training_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=errors,
        mode='lines+markers',
        name='Average Error',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Average Error",
        width=600,
        height=400
    )
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">⚡ Gate Classification with Single Layer Perceptron</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This interactive demo shows how Single Layer Perceptrons can learn to classify different logic gates.
    Select a gate type from the sidebar and experiment with different training parameters!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    gate_type = st.sidebar.selectbox(
        "Select Gate Type",
        ["AND", "OR", "AND-NOT", "XOR"],
        help="Choose which logic gate to demonstrate"
    )
    
    st.sidebar.subheader("Training Parameters")
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Controls how fast the perceptron learns"
    )
    
    epochs = st.sidebar.slider(
        "Number of Epochs",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Number of training iterations"
    )
    
    init_method = st.sidebar.radio(
        "Weight Initialization",
        ["random", "zeros"],
        help="How to initialize weights and bias"
    )
    
    # Get gate data
    X, y, description = get_gate_data(gate_type)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f'<h2 class="gate-header">{gate_type} Gate</h2>', unsafe_allow_html=True)
        st.write(description)
        
        # Truth table
        st.subheader("📊 Truth Table")
        truth_table = create_truth_table_display(X, y, gate_type)
        st.dataframe(truth_table, use_container_width=True)
        
        # Training button
        if st.button(f"🚀 Train {gate_type} Gate Perceptron", type="primary"):
            with st.spinner("Training perceptron..."):
                # Create and train perceptron
                perceptron = PerceptronGate(gate_type)
                perceptron.train(X, y, learning_rate, epochs, init_method)
                
                # Store in session state
                st.session_state[f'perceptron_{gate_type}'] = perceptron
                st.session_state[f'trained_{gate_type}'] = True
    
    with col2:
        # Check if we have a trained perceptron
        if f'perceptron_{gate_type}' in st.session_state and st.session_state.get(f'trained_{gate_type}', False):
            perceptron = st.session_state[f'perceptron_{gate_type}']
            
            # Test the perceptron
            st.subheader("🧪 Test Results")
            predictions = perceptron.predict(X)
            accuracy = np.mean(predictions == y) * 100
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Input 1': X[:, 0],
                'Input 2': X[:, 1],
                'Target': y,
                'Predicted': predictions,
                'Correct': ['✅' if p == t else '❌' for p, t in zip(predictions, y)]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Display accuracy
            if accuracy == 100:
                st.markdown(f'<div class="success-box">🎉 Perfect Accuracy: {accuracy:.1f}%</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">⚠️ Accuracy: {accuracy:.1f}%</div>', 
                           unsafe_allow_html=True)
            
            # Show final weights and bias
            st.subheader("🔧 Final Parameters")
            col_w1, col_w2, col_b = st.columns(3)
            with col_w1:
                st.metric("Weight 1", f"{perceptron.weights[0]:.3f}")
            with col_w2:
                st.metric("Weight 2", f"{perceptron.weights[1]:.3f}")
            with col_b:
                st.metric("Bias", f"{perceptron.bias[0]:.3f}")
        
        else:
            st.info("👆 Click the training button to see results!")
    
    # Full width visualizations
    if f'perceptron_{gate_type}' in st.session_state and st.session_state.get(f'trained_{gate_type}', False):
        perceptron = st.session_state[f'perceptron_{gate_type}']
        
        st.markdown("---")
        
        # Create two columns for plots
        viz_col1, viz_col2 = st.columns([1, 1])
        
        with viz_col1:
            st.subheader("📈 Decision Boundary")
            decision_fig = plot_decision_boundary(perceptron, X, y, gate_type)
            st.plotly_chart(decision_fig, use_container_width=True)
        
        with viz_col2:
            st.subheader("📉 Training Progress")
            training_fig = plot_training_progress(perceptron)
            if training_fig:
                st.plotly_chart(training_fig, use_container_width=True)
        
        # Gate-specific Q&A section
        st.markdown("---")
        st.subheader("❓ Analysis Questions & Answers")
        
        if gate_type == "AND":
            st.markdown("""
            **Question:** How do the weights and bias values change during training for the AND gate?
            
            **Answer:** During training of the AND gate, the weights and bias values get updated or changed as per the following formula:
            - `weights += learning_rate * error * xi`
            - `bias += learning_rate * error`
            
            **Question:** Can the perceptron successfully learn the AND logic with a linear decision boundary?
            
            **Answer:** Yes it can as we can draw a line which clearly separates the three false points from the one true point as per the truth table of the AND gate.
            """)
        
        elif gate_type == "OR":
            st.markdown("""
            **Question:** What changes in the perceptron's weights are necessary to represent the OR gate logic?
            
            **Answer:** The perceptron's weights remain positive in order to represent the OR gate logic whereas the bias gets adjusted to be negative.
            
            **Question:** How does the linear decision boundary look for the OR gate classification?
            
            **Answer:** The boundary line has three true points above it and one false point below it and it has a negative slope.
            """)
        
        elif gate_type == "AND-NOT":
            st.markdown("""
            **Question:** What is the perceptron's weight configuration after training for the AND-NOT gate?
            
            **Answer:** We see that the weight for Input1 is positive while the weight for Input2 is negative.
            
            **Question:** How does the perceptron handle cases where both inputs are 1 or 0?
            
            **Answer:** It detects them as false (0).
            """)
        
        elif gate_type == "XOR":
            st.markdown("""
            **Question:** Why does the Single Layer Perceptron struggle to classify the XOR gate?
            
            **Answer:** The SLP struggles to correctly classify the XOR gate as we cannot draw a straight line which clearly separates the true and false outputs, hence it is a non linear problem. By observing the weight and bias values while training we see that the perceptron will oscillate its weights and bias during training but will never converge to a solution that correctly classifies all inputs.
            
            **Question:** What modifications can be made to the neural network model to handle the XOR gate correctly?
            
            **Answer:** We can modify the Single Layer Perceptron to a Multi Layer Perceptron.
            """)
    
    # Information section
    st.markdown("---")
    st.subheader("📚 About Logic Gates")
    
    gate_info = {
        "AND": {
            "formula": "Output = Input1 AND Input2",
            "characteristics": "Linearly separable, easy to learn",
            "use_case": "Used when all conditions must be true"
        },
        "OR": {
            "formula": "Output = Input1 OR Input2", 
            "characteristics": "Linearly separable, easy to learn",
            "use_case": "Used when at least one condition must be true"
        },
        "AND-NOT": {
            "formula": "Output = Input1 AND (NOT Input2)",
            "characteristics": "Linearly separable, requires negative weight for Input2",
            "use_case": "Used for conditional logic with exclusion"
        },
        "XOR": {
            "formula": "Output = Input1 XOR Input2",
            "characteristics": "NOT linearly separable, cannot be learned by single perceptron",
            "use_case": "Used for exclusive conditions, requires multi-layer networks"
        }
    }
    
    info_cols = st.columns(4)
    for i, (gate, info) in enumerate(gate_info.items()):
        with info_cols[i]:
            selected = "🔵" if gate == gate_type else "⚪"
            st.markdown(f"### {selected} {gate}")
            st.write(f"**Formula:** {info['formula']}")
            st.write(f"**Characteristics:** {info['characteristics']}")
            st.write(f"**Use Case:** {info['use_case']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🧠 Built with Streamlit | Neural Networks Lab Demo</p>
    <p>Experiment with different parameters to understand how perceptrons learn!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()