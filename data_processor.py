import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict


class KANDataProcessor1:
    """Processes gene expression data for KAN model training."""

    def __init__(self):
        self.gene_data = {}
        self.sample_names = None
        self.related_genes = {}

    def prepare_training_data(
        self, expression_file: Path, network_file: Path, target_gene: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prepares training data for a target gene.

        Args:
            expression_file: Path to h5ad expression data file
            network_file: Path to network TSV file
            target_gene: Name of target gene

        Returns:
            Tuple of (input matrix, target values, model config)
        """
        # Load and process data
        adata = sc.read_h5ad(expression_file)
        expr_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        gene_names = adata.var_names.tolist()
        self.sample_names = adata.obs_names.tolist()

        # Get related genes from network
        network_df = pd.read_csv(network_file, sep="\t")
        related_genes = self._get_related_genes(network_df, target_gene, gene_names)
        self.related_genes[target_gene] = related_genes

        # Prepare input matrix and target vector
        X, y = self._prepare_matrices(
            expr_matrix, gene_names, target_gene, related_genes
        )

        # Create model config
        config = {
            "width": [X.shape[1], 1, 1],
            # "width": X.shape[1],
            "grid": 4,
            "k": 3,
            "seed": 42,
            # "feature_names": related_genes,
        }
        # config["feature_names"] = related_genes
        return X.astype(np.float32), y.astype(np.float32), config

    def _get_related_genes(
        self, network_df: pd.DataFrame, target_gene: str, gene_names: List[str]
    ) -> List[str]:
        """Gets list of genes related to target gene from network."""
        source_col, target_col = network_df.columns[:2]
        related = network_df[
            (network_df[source_col] == target_gene)
            | (network_df[target_col] == target_gene)
        ]

        genes = []
        for _, row in related.iterrows():
            gene = (
                row[target_col] if row[source_col] == target_gene else row[source_col]
            )
            if gene in gene_names:
                genes.append(gene)

        return genes

    def _prepare_matrices(
        self,
        expr_matrix: np.ndarray,
        gene_names: List[str],
        target_gene: str,
        related_genes: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares input and target matrices."""
        target_idx = gene_names.index(target_gene)
        related_indices = [gene_names.index(gene) for gene in related_genes]

        X = expr_matrix[:, related_indices]
        y = expr_matrix[:, target_idx]

        return X, y

    def get_related_genes(self, target_gene: str) -> List[str]:
        """Returns list of genes related to target gene."""
        return self.related_genes.get(target_gene, [])
