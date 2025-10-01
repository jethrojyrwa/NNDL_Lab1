# Gate Classification Demo with Streamlit

This interactive Streamlit application demonstrates the implementation of Single Layer Perceptrons for classifying different logic gates (AND, OR, AND-NOT, and XOR).

## Features

- **Interactive Training**: Train perceptrons with customizable parameters
- **Visual Decision Boundaries**: See how the perceptron separates different classes
- **Training Progress**: Monitor the learning process with real-time plots
- **Comprehensive Results**: View accuracy, predictions, and model parameters
- **Educational Content**: Learn about different gate types and their characteristics

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the directory containing the files
2. Run the Streamlit app:
```bash
streamlit run gate_classification_demo.py
```

3. Open your web browser and go to the displayed local URL (usually `http://localhost:8501`)

## How to Use

1. **Select a Gate Type**: Choose from AND, OR, AND-NOT, or XOR gates in the sidebar
2. **Adjust Parameters**: Modify learning rate, epochs, and weight initialization method
3. **Train the Model**: Click the training button to start the learning process
4. **Analyze Results**: View the truth table, test results, decision boundary, and training progress
5. **Experiment**: Try different parameters to see how they affect learning

## Key Insights

- **AND, OR, AND-NOT Gates**: These are linearly separable and can be successfully learned by a single perceptron
- **XOR Gate**: This is NOT linearly separable and cannot be correctly classified by a single perceptron, demonstrating the limitation of single-layer networks

## Educational Value

This demo helps understand:
- How perceptrons learn through weight updates
- The concept of linear separability
- Why some problems require multi-layer networks
- The importance of proper parameter selection
- Visual interpretation of decision boundaries

## Technical Implementation

The application uses:
- **NumPy**: For numerical computations and matrix operations
- **Streamlit**: For creating the interactive web interface
- **Plotly**: For creating interactive visualizations
- **Pandas**: For data manipulation and display
- **Matplotlib**: For additional plotting capabilities

Enjoy exploring the fascinating world of neural networks and logic gate classification!