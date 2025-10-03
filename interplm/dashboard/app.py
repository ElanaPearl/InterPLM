import os
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from interplm.dashboard.dashboard_cache import DashboardCache
import numpy as np
import pandas as pd
import streamlit as st
import torch

from interplm.dashboard.colors import get_structure_palette_and_colormap
from interplm.dashboard.feature_activation_vis import (
    plot_activation_scatter,
    plot_activations_for_single_feat,
    plot_structure_scatter,
    plot_umap_scatter,
    visualize_protein_feature,
)
from interplm.dashboard.help_notes import help_notes
from interplm.dashboard.view_structures import view_single_protein
from interplm.data_processing.utils import fetch_uniprot_sequence
from interplm.embedders import get_embedder
from interplm.utils import get_device


@dataclass
class DashboardState:
    """Holds the state and configuration for the dashboard"""

    layer: int
    feature_id: int
    feature_activation_range: str
    n_proteins_to_show: int
    add_highlight: bool
    custom_uniprot_ids: List[str] | None
    custom_sequences: List[str] | None
    show_proteins: bool


class ProteinFeatureVisualizer:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.dashboard_data = self._load_data(cache_dir)
        self.device = get_device()

        # Initialize embedder for ESM models
        self.esm_embedder = None
        # Initialize embedder if model_name and model_type are provided
        if hasattr(self.dashboard_data, 'model_name') and hasattr(self.dashboard_data, 'model_type'):
            if self.dashboard_data.model_name and self.dashboard_data.model_type:
                # Get default device, but avoid MPS due to bugs that corrupt embeddings
                embedder_device = get_device()
                if embedder_device == "mps":
                    embedder_device = "cpu"

                # Use the model_name from dashboard metadata (e.g., 'esm', 'progen2')
                embedder_type = self.dashboard_data.model_name
                model_name = self.dashboard_data.model_type

                # For ESM models, prepend 'facebook/' if not already present
                if embedder_type == 'esm' and not model_name.startswith('facebook/'):
                    model_name = f"facebook/{model_name}"

                self.esm_embedder = get_embedder(
                    embedder_type,
                    model_name=model_name,
                    device=embedder_device
                )


    @property
    def protein_id_col(self):
        """Get the protein ID column name for the current metadata type."""
        metadata = self.dashboard_data.protein_metadata
        if hasattr(metadata, 'uniprot_id_col'):
            return metadata.uniprot_id_col
        else:
            return 'protein_id'

    @property
    def sequence_col(self):
        """Get the sequence column name for the current metadata type."""
        metadata = self.dashboard_data.protein_metadata
        if hasattr(metadata, 'sequence_col'):
            return metadata.sequence_col
        else:
            return 'sequence'


    @staticmethod
    @st.cache_resource
    def _load_data(cache_dir: Path):
        """Load and cache the dashboard data

        Args:
            cache_dir: Full path to the dashboard cache directory
        """
        dash_data = DashboardCache.load_cache(cache_dir=cache_dir)

        return dash_data

    def select_feature(self, layer):
        layer_data = self.dashboard_data.data[layer]
        n_features = layer_data["SAE"].dict_size

        # Initialize session state for feature_id if not exists
        if f"feature_id_{layer}" not in st.session_state:
            st.session_state[f"feature_id_{layer}"] = layer_data.get(
                "Default feature", 0
            )

        # Get available features to sample from
        feats_to_sample = []
        if "Sig_concepts_per_feature" in layer_data.keys():
            sig_concepts = (
                layer_data["Sig_concepts_per_feature"]
                .query("f1_per_domain > 0.5")["feature"]
                .unique()
            )
            if len(sig_concepts) > 0:
                feats_to_sample.extend(sig_concepts)
        if "LLM Autointerp" in layer_data.keys():
            feats_to_sample.extend(
                layer_data["LLM Autointerp"].query("Correlation > 0.5").index
            )
        if len(feats_to_sample) == 0:
            feats_to_sample = list(range(n_features))

        # Create random feature button below the number input
        if st.sidebar.button("Select random feature"):
            # Update session state with random feature
            st.session_state[f"feature_id_{layer}"] = int(
                np.random.choice(feats_to_sample)
            )

        # Number input that uses and updates session state
        feature_id = st.sidebar.number_input(
            f"Or specify SAE feature number",
            min_value=0,
            max_value=n_features - 1,
            step=1,
            value=st.session_state[f"feature_id_{layer}"],
            key=f"feature_input_{layer}",
            help=f"Enter a specific feature ID to explore (0 - {n_features-1:,})",
        )

        # Update session state if number input changes
        st.session_state[f"feature_id_{layer}"] = feature_id

        return feature_id

    def setup_sidebar(self) -> DashboardState:
        """Configure sidebar controls and return dashboard state"""
        st.sidebar.markdown(help_notes["overall"], unsafe_allow_html=True)
        st.sidebar.markdown(
            f"## Select {self.dashboard_data.model_type} Layer and Feature",
            help=help_notes["select_esm_layer"],
        )
        available_layers = sorted(self.dashboard_data.layers)
        if len(available_layers) == 0:
            st.error("No data found. Please check the cache and try again.")
        elif len(available_layers) == 1:
            layer = available_layers[0]
        else:
            layer = st.sidebar.selectbox(
                "Select model embedding layer",
                available_layers,
                index=3 if 3 in available_layers else 0,
            )

        dash_data = self.dashboard_data.data[layer]
        feature_id = self.select_feature(layer)
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "## Visualize Feature Activation on Proteins",
            help=help_notes["vis_sidebar"],
        )

        st.sidebar.markdown(
            "Proteins are shown by default. Customize selection below or click 'Refresh Visualizations' after changes:"
        )
        show_proteins = st.sidebar.button("Refresh Visualizations")

        # Specify activation range
        st.sidebar.markdown(
            "### a) Select proteins by activation range",
            help="Select proteins based on how strongly they activate the feature.",
        )
        quantiles = [
            i
            for i, sublist in dash_data["Per_feature_quantile_examples"][
                feature_id
            ].items()
            if len(sublist) > 0
        ]
        available_ranges = ["Max"] + quantiles
        feature_activation_range = st.sidebar.selectbox(
            "Select activation group", available_ranges, index=0
        )

        # Number of proteins to visualize
        max_prot_ids = self._get_protein_ids(
            dash_data, feature_id, feature_activation_range
        )
        n_possible = min(10, len(max_prot_ids))
        default_n = min(5, n_possible)
        if n_possible == 0:
            st.sidebar.warning("No proteins found in this activation range.")
            n_proteins = 0
        else:
            n_proteins = st.sidebar.slider(
                "Number of proteins to visualize",
                0,
                n_possible,
                default_n,
                help="Maximum of 10 proteins can be shown",
            )

        # Custom protein selection
        st.sidebar.markdown(
            "### b) Enter custom UniProt IDs", help=help_notes["uniprot"]
        )
        custom_ids = st.sidebar.text_area(
            "Enter Uniprot IDs",
            "",
            placeholder="Add each Uniprot ID to a new line. These are slower as they fetch data from UniProt.",
        )

        # Custom sequence selection
        st.sidebar.markdown(
            "### c) Enter custom protein sequences",
            help="Note: Custom sequences will not show the protein structure.",
        )
        custom_sequences = st.sidebar.text_area(
            "Enter protein sequences",
            "",
            placeholder="Add each protein sequence to a new line",
        )

        add_highlight = st.sidebar.checkbox(
            "Highlight high activation residues",
            value=False,
            help="Highlight residues with activations in the top 5% of the range",
        )

        if custom_ids and custom_sequences:
            st.warning(
                "Please select either Uniprot IDs or custom sequences, not both."
            )
            st.stop()

        elif custom_ids:
            custom_ids = [id.strip() for id in custom_ids.split("\n") if id.strip()]
        elif custom_sequences:
            custom_sequences = [
                seq.strip() for seq in custom_sequences.split("\n") if seq.strip()
            ]

        return DashboardState(
            layer=layer,
            feature_id=feature_id,
            feature_activation_range=feature_activation_range,
            n_proteins_to_show=n_proteins,
            add_highlight=add_highlight,
            custom_uniprot_ids=custom_ids,
            custom_sequences=custom_sequences,
            show_proteins=show_proteins,
        )

    def visualize_proteins(self, state: DashboardState):
        """Visualize selected proteins with their feature activations"""
        dash_data = self.dashboard_data.data[state.layer]

        try:
            if state.custom_uniprot_ids:
                proteins_to_viz = self._get_custom_proteins(state.custom_uniprot_ids)

                if proteins_to_viz.empty:
                    st.warning(
                        "No valid UniProt IDs found. Please check the IDs and try again."
                    )
                    return
            elif state.custom_sequences:
                proteins_to_viz = pd.DataFrame(
                    {
                        self.protein_id_col: [
                            f"Custom Sequence {i+1}"
                            for i in range(len(state.custom_sequences))
                        ],
                        self.sequence_col: state.custom_sequences,
                    }
                )
            else:
                proteins_to_viz = self._get_proteins_by_activation(
                    dash_data,
                    state.feature_id,
                    state.feature_activation_range,
                    state.n_proteins_to_show,
                )
                if proteins_to_viz.empty:
                    st.warning("No proteins found that activate this feature.")
                    return

            color_range = self._get_color_range(state.feature_activation_range)
            structure_colormap_fn, palette_to_viz = get_structure_palette_and_colormap(
                color_range
            )

            self._render_protein_visualizations(
                proteins_to_viz,
                dash_data,
                state.feature_id,
                state.layer,
                state.add_highlight,
                structure_colormap_fn,
                palette_to_viz,
                is_custom_seq=bool(state.custom_sequences),
            )
        except Exception as e:
            st.error(f"Error visualizing proteins: {str(e)}")
            st.error(traceback.format_exc())

    def display_feature_statistics(self, layer: int, feature_id: int):
        """Display feature-wide statistics section"""
        dash_data = self.dashboard_data.data[layer]

        # Extract layer number from layer name (e.g., "layer_4" -> "4")
        layer_num = layer.split('_')[1] if '_' in str(layer) else str(layer)

        st.header(
            f"Metrics on all SAE features from {self.dashboard_data.model_type} layer {layer_num}",
            help=help_notes["metrics"],
        )
        st.markdown(
            f"**(Highlighting selected feature <span style='color: #00DDFF'>f/{feature_id}**</span>)",
            unsafe_allow_html=True,
            help="Feature and layer selection can be changed in the sidebar.",
        )

        # Check which plots are available
        has_structure = "Structure_features" in dash_data
        has_umap = "UMAP" in dash_data
        has_concepts = "Sig_concepts_per_feature" in dash_data

        # Count available visualizations
        available_plots = []
        available_plots.append(("activation_consistency", self._plot_activation_consistency))
        if has_structure:
            available_plots.append(("structure", self._plot_structure_features))
        if has_umap:
            available_plots.append(("umap", self._plot_umap))
        if has_concepts:
            available_plots.append(("concepts", self._display_swissprot_concepts))

        # Create columns based on number of available plots
        num_cols = len(available_plots)
        if num_cols == 0:
            st.warning("No feature statistics available")
        elif num_cols == 1:
            # Single column - full width
            self._plot_activation_consistency(dash_data, feature_id)
        else:
            # Multiple columns
            cols = st.columns(num_cols)
            for idx, (plot_type, plot_func) in enumerate(available_plots):
                with cols[idx]:
                    if plot_type == "activation_consistency":
                        plot_func(dash_data, feature_id)
                    elif plot_type == "structure":
                        plot_func(dash_data, feature_id)
                    elif plot_type == "umap":
                        plot_func(dash_data, feature_id)
                    elif plot_type == "concepts":
                        plot_func(dash_data, feature_id, layer)

        st.markdown("---")
        st.header(f"Details on f/{feature_id}", help=help_notes["feature_details"])

        self._display_feature_act_dist_and_concepts(dash_data, feature_id)

        st.markdown("---")

    def _display_feature_act_dist_and_concepts(self, dash_data: Dict, feature_id: int):
        # Check whether "LLM Autointerp" in dash_data.keys()
        if (
            "LLM Autointerp" in dash_data.keys()
            and feature_id in dash_data["LLM Autointerp"].index
        ):
            col3, col1, col2 = st.columns(3)
            with col3:
                # write dash_data["LLM Autointerp"].loc[feature_id]
                description_score = (
                    f"{dash_data['LLM Autointerp'].loc[feature_id]['Correlation']:.2f}"
                )
                st.subheader(
                    f"**Language Model Summary for f/{feature_id} (score={description_score})**",
                    help=help_notes["autointerp"],
                )
                st.write(f"{dash_data['LLM Autointerp'].loc[feature_id]['Summary']}")
        else:
            col1, col2 = st.columns(2)
        with col1:
            st.subheader(
                f"**Feature Activation Distribution for f/{feature_id}**",
                help=help_notes["act_distribution"],
            )
            plot_of_feat_acts = plot_activations_for_single_feat(
                dash_data["SAE_features"], feature_id
            )
            if plot_of_feat_acts is not None:
                st.plotly_chart(
                    plot_of_feat_acts,
                    use_container_width=True,
                )
            else:
                st.write("No activations found for this feature in random sample.")

            with col2:
                # Use Sig_concepts_per_feature from dashboard data
                if "Sig_concepts_per_feature" in dash_data:
                    concepts_for_feat = dash_data["Sig_concepts_per_feature"][
                        dash_data["Sig_concepts_per_feature"]["feature"] == feature_id
                    ].copy()

                    if not concepts_for_feat.empty:
                        # Filter by F1 threshold
                        concepts_for_feat = concepts_for_feat[concepts_for_feat["f1_per_domain"] > 0.5]

                        # Keep only best threshold per concept (highest F1)
                        concepts_for_feat = concepts_for_feat.sort_values("f1_per_domain", ascending=False)
                        concepts_for_feat = concepts_for_feat.drop_duplicates("concept", keep="first")

                        # Format for display
                        display_df = concepts_for_feat[["concept", "f1_per_domain", "precision", "recall", "threshold_pct"]].copy()
                        display_df.columns = ["Concept", "F1", "Precision", "Recall", "Threshold %"]
                        display_df.set_index("Concept", inplace=True)
                        st.write(display_df)

                        if len(concepts_for_feat) == 0:
                            st.write("No concepts found with F1 > 0.5 for this feature.")
                    else:
                        st.write("No concepts found for this feature.")
                elif (
                    "Sig_concepts_per_feature" in dash_data.keys()
                    and not dash_data["Sig_concepts_per_feature"].empty
                ):
                    # Fall back to old dashboard data format
                    concepts_for_feat = (
                        dash_data["Sig_concepts_per_feature"]
                        .query(f"f1_per_domain > 0.2 & feature == {feature_id}")
                        .sort_values("f1_per_domain", ascending=False)
                    )

                    if len(concepts_for_feat) > 0:
                        concepts_for_feat = concepts_for_feat.drop_duplicates(
                            "concept", keep="first"
                        )
                        concepts_for_feat = concepts_for_feat[
                            [
                                "concept",
                                "threshold_pct",
                                "f1_per_domain",
                                "precision",
                                "recall_per_domain",
                                "tp",
                                "tp_per_domain",
                                "fp",
                            ]
                        ]
                        concepts_for_feat.rename(
                            columns={
                                "concept": "Concept",
                                "f1_per_domain": "F1",
                                "precision": "Precision",
                                "recall_per_domain": "Recall",
                                "tp": "True Positives (per AA)",
                                "tp_per_domain": "True Positives (per Domain)",
                                "threshold_pct": "Threshold",
                            },
                            inplace=True,
                        )
                        concepts_for_feat.set_index("Concept", inplace=True)
                        st.write(concepts_for_feat)
                    else:
                        st.write("No Swiss-Prot concepts found for this feature.")
                else:
                    st.write("No concepts found for this feature.")

    def _display_top_features_per_concept(self, dash_data: Dict):
        """Display a table showing the top feature for each concept across all features."""
        st.subheader("🧬 Top Features per Concept")
        st.caption("*Showing the best-matching feature for each biological concept (F1 > 0.5)*")

        # Check if concept data exists
        if "Sig_concepts_per_feature" not in dash_data:
            st.info("No concept association data available for this layer")
            return

        concept_results = dash_data["Sig_concepts_per_feature"]

        # Get the top feature for each concept (highest F1 score), filtered by threshold
        top_per_concept = (
            concept_results[concept_results["f1_per_domain"] > 0.5]
            .sort_values("f1_per_domain", ascending=False)
            .drop_duplicates("concept", keep="first")
            .sort_values("f1_per_domain", ascending=False)
        )

        # Format for display
        display_data = []
        for _, row in top_per_concept.iterrows():
            display_data.append({
                "Concept": row["concept"],
                "Top Feature": f"f/{int(row['feature'])}",
                "F1 (per domain)": f"{row['f1_per_domain']:.4f}",
                "Precision": f"{row['precision']:.4f}",
                "Recall": f"{row['recall']:.4f}",
            })

        display_df = pd.DataFrame(display_data)

        # Display as interactive table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400,  # Fixed height with scrolling
        )

    def _display_concept_analysis(self, dash_data: Dict, feature_id: int):
        """Display concept association analysis results for a specific feature."""
        st.subheader("🧬 Functional Annotation Concept Analysis")

        # Check if concept data exists
        if "Sig_concepts_per_feature" not in dash_data:
            st.info("No concept association data available for this layer")
            return

        concept_results = dash_data["Sig_concepts_per_feature"]

        # Filter results for this feature
        feature_concepts = concept_results[
            concept_results["feature"] == feature_id
        ].copy()

        if feature_concepts.empty:
            st.info(f"No concept associations found for feature {feature_id}")
            return

        # Sort by F1 score descending
        feature_concepts = feature_concepts.sort_values("f1_per_domain", ascending=False)

        # Show summary statistics
        st.markdown(f"**Found {len(feature_concepts)} concept associations for this feature**")
        st.caption("*Showing concepts matched using F1 score between feature activations and UniProtKB annotations*")

        # Create display table
        display_data = []
        for _, row in feature_concepts.iterrows():
            display_data.append({
                "Concept": row["concept"],
                "Description": row["description"] if pd.notna(row.get("description")) else row["concept"],
                "F1 Score": f"{row['f1_per_domain']:.4f}",
                "Precision": f"{row['precision']:.4f}",
                "Recall": f"{row['recall']:.4f}",
                "Threshold": f"{row['threshold']:.2f}",
            })

        display_df = pd.DataFrame(display_data)

        # Display as interactive table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Concept": st.column_config.TextColumn(
                    "Concept",
                    width="medium",
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    width="large",
                ),
                "F1 Score": st.column_config.TextColumn(
                    "F1 Score",
                    help="Harmonic mean of precision and recall",
                ),
                "Precision": st.column_config.TextColumn(
                    "Precision",
                    help="What fraction of feature activations match the concept",
                ),
                "Recall": st.column_config.TextColumn(
                    "Recall",
                    help="What fraction of concept annotations are captured by the feature",
                ),
                "Threshold": st.column_config.TextColumn(
                    "Threshold",
                    help="Activation threshold used for matching",
                ),
            }
        )

    def _get_protein_ids(
        self, dash_data: Dict, feature_id: int, activation_range: str
    ) -> List[str]:
        """Get protein IDs based on activation range"""
        if activation_range == "Max":
            return list(dash_data["Per_feature_max_examples"][feature_id])
        return list(
            dash_data["Per_feature_quantile_examples"][feature_id][activation_range]
        )

    def _get_color_range(self, activation_range: str) -> Tuple[float, float, float]:
        """Get color range based on activation range"""
        if activation_range == "Max":
            return (0, 0.4, 0.85)
        try:
            range_value = float(activation_range[0])
            return (0, range_value / 2, range_value)
        except (IndexError, ValueError):
            return (0, 0.2, 0.4)  # Default fallback

    def _get_custom_proteins(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Get protein data for custom Uniprot IDs"""
        metadata = []
        for uniprot_id in uniprot_ids:
            protein_metadata = fetch_uniprot_sequence(uniprot_id)
            if protein_metadata:
                metadata.append(protein_metadata)
        return pd.DataFrame(metadata)

    def _get_proteins_by_activation(
        self, dash_data: Dict, feature_id: int, activation_range: str, n_proteins: int
    ) -> pd.DataFrame:
        """Get protein data based on activation range"""
        protein_ids = self._get_protein_ids(dash_data, feature_id, activation_range)
        # TODO: handle this earlier
        protein_ids = [id.upper() for id in protein_ids]
        if not protein_ids:
            return pd.DataFrame()  # Return empty DataFrame if no proteins found
        
        # Get protein metadata from UniProt
        metadata = self.dashboard_data.protein_metadata
        # Convert to uppercase to match the index (metadata converts IDs to uppercase)
        protein_ids_upper = [pid.upper() for pid in protein_ids[:n_proteins]]
        return metadata.data.loc[protein_ids_upper].reset_index()

    def _plot_structure_features(self, dash_data: Dict, feature_id: int):
        """Plot structure features scatter plot"""
        if "Structure_feats" not in dash_data:
            return
        struct_data = dash_data["Structure_feats"].set_index("feat")
        plot = plot_structure_scatter(
            df=struct_data,
            title="",
            feature_to_highlight=(
                feature_id if feature_id in struct_data.index else None
            ),
        )
        struct_v_seq_help = (
            "When a feature activates on multiple amino acids in a protein,"
            "are the activated positions close nearby eachother in 3D space? Are they nearby in "
            "the sequence? This compares these two ways of looking at features activation. If you "
            "look features with structures:seq ratios, they are often interesting"
        )
        st.subheader("**Structural vs Sequential**", help=struct_v_seq_help)
        st.plotly_chart(
            plot, use_container_width=True
        )

    def _plot_activation_consistency(self, dash_data: Dict, feature_id: int):
        """Plot activation consistency scatter plot"""
        if "Per_feature_statistics" not in dash_data:
            return
        stats = dash_data["Per_feature_statistics"]
        plot = plot_activation_scatter(
            x_value=stats["Per_prot_frequency_of_any_activation"],
            y_value=stats["Per_prot_pct_activated_when_present"],
            title="",
            xaxis_title="% of proteins with activation",
            yaxis_title="Avg % activated when present",
            feature_to_highlight=feature_id,
        )
        st.subheader(
            "**Feature Activation Frequencies**",
            help="Shows the consistency of feature activation across and within proteins.",
        )
        st.plotly_chart(
            plot, use_container_width=True
        )

    def _plot_umap(self, dash_data: Dict, feature_id: int):
        """Plot UMAP visualization"""
        if "UMAP" not in dash_data:
            return
        umap_data = dash_data["UMAP"].reset_index().rename(columns={"index": "Feature"})
        plot = plot_umap_scatter(
            umap_data,
            feature_to_highlight=(
                feature_id if feature_id in umap_data["Feature"] else None
            ),
            title="",
        )
        st.subheader(
            "**UMAP of Feature Values**",
            help="All features in layer visualized in 2D based on a UMAP of their dictionary values. Coloring based on cluster assignment.",
        )
        st.plotly_chart(
            plot,
            use_container_width=True,
        )

    def _display_swissprot_concepts(self, dash_data: Dict, feature_id: int, layer: int):
        """Display SwissProt concepts table"""
        if "Sig_concepts_per_feature" not in dash_data:
            return
        concepts = (
            dash_data["Sig_concepts_per_feature"]
            .query("tp_per_domain >= 2 or tp >= 2")
            .sort_values(["f1_per_domain", "recall_per_domain", "tp"], ascending=False)
            .drop_duplicates(["concept"], keep="first")
        )

        # Extract layer number from layer name (e.g., "layer_4" -> "4")
        layer_num = layer.split('_')[1] if '_' in str(layer) else str(layer)

        st.subheader(
            f"**Concepts Identified in Layer {layer_num}**",
            help="Concepts are defined based on Swiss-Prot annotations. For each concept identified in the SAE features, we list one example feature that activates on this concept. Details on concept-feature pairing is described in the InterPLM paper.",
        )
        display_cols = {
            "concept": "Concept",
            "feature": "Feature",
            "f1_per_domain": "F1",
            "precision": "Precision",
            "recall_per_domain": "Recall",
            "tp": "True Positives (per AA)",
        }

        st.dataframe(
            concepts[list(display_cols.keys())]
            .rename(columns=display_cols)
            .set_index("Concept"),
            height=300,
        )

    def _render_protein_visualizations(
        self,
        proteins: pd.DataFrame,
        dash_data: Dict,
        feature_id: int,
        layer: str,
        add_highlight: bool,
        colormap_fn,
        palette_to_viz,
        is_custom_seq: bool = False,
    ):
        """Render visualizations for selected proteins"""
        # Extract layer number from layer name (e.g., "layer_3" -> 3)
        layer_num = int(layer.split('_')[1])

        for idx, (_, protein) in enumerate(proteins.iterrows()):
            try:
                # Get feature activations (ESM only)
                if self.esm_embedder is None:
                    raise ValueError("ESM embedder not initialized")
                embeddings_np = self.esm_embedder.embed_single_sequence(
                    sequence=protein[self.sequence_col],
                    layer=layer_num
                )
                # Convert to tensor for SAE
                embeddings = torch.from_numpy(embeddings_np).to(self.device)
                features = (
                    dash_data["SAE"]
                    .encode_feat_subset(
                        x=embeddings, feat_list=[feature_id], normalize_features=True
                    )
                    .cpu()
                    .numpy()
                    .flatten()
                )

                # Normalize features by dividing by the max value for this protein
                # This is ONLY used for structure visualization (not sequence)
                max_activation = np.max(features)
                if max_activation > 1e-8:  # Avoid division by zero
                    features_normalized = features / max_activation
                else:
                    features_normalized = features  # Keep original if all are essentially zero

                # Display protein header
                if is_custom_seq:
                    st.subheader(f"Custom Sequence {idx+1}")
                else:
                    protein_id = protein[self.protein_id_col]

                    # Get protein name if available
                    protein_name = ""
                    if hasattr(self.dashboard_data, 'protein_metadata') and hasattr(self.dashboard_data.protein_metadata, 'protein_name_col'):
                        protein_name_col = self.dashboard_data.protein_metadata.protein_name_col
                        if protein_name_col in protein.index:
                            protein_name = protein[protein_name_col]

                    # Display subtitle with protein info and UniProtKB link
                    if protein_name:
                        st.markdown(f"**[{protein_id}](https://www.uniprot.org/uniprotkb/{protein_id})**: {protein_name}")
                    else:
                        st.markdown(f"**[{protein_id}](https://www.uniprot.org/uniprotkb/{protein_id})**")

                if idx == 0:
                    col1, col2, col3 = st.columns([3, 3, 1])
                    with col3:
                        st.plotly_chart(
                            palette_to_viz,
                            use_container_width=True,
                        )
                else:
                    col1, col2 = st.columns([3, 5])

                # Display visualizations
                with col1:
                    st.plotly_chart(
                        visualize_protein_feature(
                            features,  # Use raw features to compare activation levels across proteins
                            protein[self.sequence_col],
                            protein,
                            "Amino Acids",
                        ),
                        use_container_width=True,
                    )

                with col2:
                    if not is_custom_seq:
                        # Use normalized features with adjusted threshold (now that max is 1.0)
                        highlight_threshold = 0.8  # Top 20% of normalized activations
                        self._render_protein_structure(
                            protein[self.protein_id_col],
                            features_normalized,  # Use normalized features for structure
                            colormap_fn,
                            (
                                [idx for idx, val in enumerate(features_normalized) if val > highlight_threshold]
                                if add_highlight
                                else None
                            ),
                        )
            except Exception as e:
                st.error(
                    f"Error visualizing protein {protein[self.protein_id_col]}: {str(e)}"
                )
                continue

    def _render_protein_structure(
        self,
        uniprot_id: str,
        features: np.ndarray,
        colormap_fn,
        highlight_residues: Optional[List[int]] = None,
    ):
        """Render 3D protein structure visualization"""
        # Use AlphaFold structure download
        structure_html = view_single_protein(
            uniprot_id=uniprot_id,
            values_to_color=features,
            colormap_fn=colormap_fn,
            residues_to_highlight=highlight_residues,
            pymol_params={"width": 500, "height": 300},
        )
        st.components.v1.html(structure_html, height=300)


def main(cache_dir: str):
    st.set_page_config(
        layout="wide",
        page_title="InterPLM",
        page_icon="🧬",
    )

    # Load custom CSS if available (optional)
    css_paths = [
        Path(__file__).parent / ".streamlit" / "style.css",  # Relative to app.py
        Path(".streamlit/style.css"),  # Relative to cwd
    ]
    css_loaded = False
    for css_path in css_paths:
        if css_path.exists():
            with open(css_path) as f:
                css_content = f.read()
                st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            css_loaded = True
            break
    if not css_loaded:
        print(f"✗ CSS not found. Tried: {[str(p) for p in css_paths]}")

    st.title("InterPLM Feature Visualization")

    visualizer = ProteinFeatureVisualizer(cache_dir=Path(cache_dir))
    state = visualizer.setup_sidebar()

    visualizer.display_feature_statistics(state.layer, state.feature_id)

    # Show proteins by default, or when button is clicked
    if state.show_proteins or True:  # Always show unless explicitly disabled
        visualizer.visualize_proteins(state)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Full path to dashboard cache directory (e.g., $INTERPLM_DATA/dashboard_cache/walkthrough)"
    )
    args = parser.parse_args()

    main(cache_dir=args.cache_dir)
