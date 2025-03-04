# KAN-GRN: Gene Regulatory Network Inference using Kolmogorov-Arnold Networks

This project presents a new approach to the inference of Gene Regulatory Networks (GRNs) based on Kolmogorov-Arnold Networks (KANs). KANs are neural networks that utilize adaptive activation functions along edges, instead of conventional fixed activation functions along nodes, making them highly appropriate for discovering intricate relationships in biological data.

## Features

- Process gene expression data from H5AD files
- Train KAN models to predict target gene expression based on related genes
- Visualize training metrics in real-time
- Automatically extract symbolic formulas from trained models
- Analyze feature contributions to understand gene relationships

## Installation

1. Clone this repository:
```bash
git clone https://github.com/poshraj24/KAN-BioInformatics.git
cd KAN-BioInformatics
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3.1 Requirements
```
#python==3.10.0
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml
pykan
scanpy
```

## Data Preparation

The code expects data to be organized in a `Data` directory with the following files:

1. `expression_data1.h5ad`: Gene expression data in H5AD format (AnnData)
2. `JUN_interactions.tsv`: Gene interaction network in TSV format with source and target genes

If you're using different filenames or target genes, you'll need to modify the paths in `main.py`.

### Example Data Format

- **H5AD file**: Contains gene expression matrix with genes as columns and samples as rows
- **Network TSV file**: At minimum, needs two columns for source and target genes (tab-separated)

## Usage

Run the main script to train a KAN model for the JUN gene:

```bash
python main.py
```

This will:
1. Load and process gene expression data
2. Set up a KAN model
3. Train the model with early stopping
4. Evaluate the model on test data
5. Extract a symbolic formula representing the gene relationships
6. Generate visualizations of the model and training metrics
7. Analyze feature contributions

### Customization

To train a model for a different target gene as well as training parameters like optimizer, learning rate and epochs , modify the relevant section in `main.py`:

```python
X_data, y_target, config = processor.prepare_training_data(
    Path("Data/expression_data1.h5ad"), 
    Path("Data/YOUR_GENE_interactions.tsv"), 
    "YOUR_TARGET_GENE"
)
```
To define the model architecture with different width, grid size (g) and spline parameter (k), modify the relevant section in `data_processor.py`
```python
# Create model config
        config = {
            "width": [X.shape[1], 10, 1],
            # "width": X.shape[1],
            "grid": 5,
            "k": 4,
            "seed": 42,
                   }
```
## Output and Interpretation

The code generates several outputs:

1. **Training Log (`training_log.txt`)**: Contains detailed information about model configuration, training statistics, and the symbolic formula
2. **Training Metrics Plot (`training_metrics.png`)**: Visualizes loss, R², RMSE, and MAE during training
3. **Model Architecture Visualization (`model_plot.png`)**: Shows the KAN structure with weights
4. **Saved Model (`final_model.pt`)**: The trained model that can be loaded for further analysis
5. **Feature Contributions**: A visualization and analysis of which genes contribute most to the prediction. You might need to manually save the feature contribution to the `input contributions` folder.

### Symbolic Formula

One of the most valuable outputs is the symbolic formula derived from the KAN model. This formula represents the mathematical relationship between the target gene and its regulators, providing interpretable insights into the regulatory mechanisms. Further simplification of the symbolic formula is needed adhering to input feature contributions and significance analysis using partial derivatives.

## Project Structure

- `main.py`: Main script for running the model training and analysis. It defines the training parameters. 
- `data_processor.py`: Contains the `KANDataProcessor1` class for processing gene expression data. It also defines the KAN's architecture for the training. 
- `kan_model.py`: Contains model training logic, metrics, and early stopping implementation
- `visualizer.py`: Contains the `LiveVisualizer` class for plotting training metrics
- `requirements.txt`: Lists all required dependencies

### Project Folder Structure:
```
KAN-BIOINFORMATICS/
├── Data/                      # Contains expression data and interaction files
├── figures/                   # Output figures directory 
├── input_contributions/       # Analysis of gene contributions
├── KAN_Train/                 # Core implementation modules
│   ├── data_processor.py      # Data processing implementation
│   ├── kan_model.py           # KAN model implementation
│   └── visualizer.py          # Visualization utilities
├── model/                     # Model specifications and configurations
├── .gitignore                 # Git ignore configuration
├── best_model.pt              # Best model checkpoint (PyTorch)
├── experiment.ipynb           # Jupyter notebook for experiments
├── final_model.pt             # Final trained model (PyTorch)
├── gitignore.txt              # Text version of gitignore
├── main.py                    # Main execution script
├── model_plot.png             # Visualization of the KAN architecture
├── README.md                  # Project documentation
├── requirements.txt           # Required dependencies
├── training_history.png       # Training history visualization
├── training_log.txt           # Detailed training logs
└── training_metrics.png       # Performance metrics visualization
```

## Advanced Usage
### Analyzing Feature Contributions

After training, the code automatically analyzes which features (genes) contribute most to the prediction:

```python
contributions = visualize_kan_node_importance(model, related_genes)
```

This helps identify the most important regulatory relationships for your target gene.

### Symbolic Formula Extraction

The trained KAN model is automatically converted to a symbolic formula:

```python
model.auto_symbolic()
formula_tuple = model.symbolic_formula()
```

This formula can provide biological insights into the regulatory mechanisms.

## Citation

If you use this code in your research, please cite:

```
Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y., & Tegmark, M. (2024). 
KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756v4.
```
## Tutorial

Check out the [experiment notebook](experiment.ipynb) for a step-by-step tutorial.
## License

This project is licensed under the MIT License - see the LICENSE file for details.
