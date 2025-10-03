# InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders
![Example feature activation patterns](https://github.com/user-attachments/assets/fc486fea-9303-45d3-aab2-9f9ceb51ac26)

InterPLM is a toolkit for extracting, analyzing, and visualizing interpretable features from protein language models (PLMs) using sparse autoencoders (SAEs). To learn more, check out [the paper in Nature Methods](https://www.nature.com/articles/s41592-025-02836-7) ([free preprint](https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1)), or explore SAE features from every hidden layer of ESM-2-8M in our interactive dashboard, [InterPLM.ai](https://interPLM.ai).

## Key Features
- 🧬 Extract SAE features from protein language models (PLMs)
- 📊 Analyze and interpret learned features through association with protein annotations
- 🎨 Visualize feature patterns and relationships 
- 🤗 Pre-trained sparse autoencoders for ESM-2 models (8M and 650M)

## Getting Started
#### Installation
```bash
# Clone the repository
git clone https://github.com/ElanaPearl/interPLM.git
cd interPLM

# Create and activate conda environment
conda env create -f environment.yml
conda activate interplm

# Install package
pip install -e .
```

## Using Pretrained Models
We provide pretrained sparse autoencoders on HuggingFace for two ESM-2 models:

| Model | Available Layers | HuggingFace Link |
|-------|-----------------|------------------|
| ESM-2-8M | 1, 2, 3, 4, 5, 6 | [InterPLM-esm2-8m](https://huggingface.co/Elana/InterPLM-esm2-8m) |
| ESM-2-650M | 1, 9, 18, 24, 30, 33 | [InterPLM-esm2-650m](https://huggingface.co/Elana/InterPLM-esm2-650m) |

You can explore these features interactively in our pre-made dashboard at [InterPLM.ai](https://interPLM.ai).

To use a pretrained model:
```python
from interplm.sae.inference import load_sae_from_hf

# Load specific layer SAE (e.g., layer 4 from ESM-2-8M)
sae = load_sae_from_hf(plm_model="esm2-8m", plm_layer=4)

# Or for ESM-2-650M (e.g., layer 24)
sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=24)
```

## Training and Analyzing Custom SAEs: Complete Guide
**_This walks through training, analysis, and feature visualization for custom SAEs based on PLM embeddings. The code is primarily set up for ESM-2 embeddings, but can easily be adapted to embeddings from any PLM (see [Adding Your Own PLM](interplm/embedders/README.md))._**

### 0. Environment setup
Set the `INTERPLM_DATA` environment variable to establish the base directory for all data paths in this walkthrough (any downloaded .fasta files and ESM-2 embeddings created). If you don't want to use an environment variable, just replace `INTERPLM_DATA` with your path of choice throughout the walkthrough.
```bash
# For zsh (replace with .bashrc or preferred shell)
echo 'export INTERPLM_DATA="$HOME/your/preferred/path"' >> ~/.zshrc
source ~/.zshrc
```

### 1. Extract PLM embeddings for training data

**Set the layer to analyze:**
```bash
# Choose which layer to extract and analyze (4 is middle layer for ESM-2-8M)
export LAYER=4
```

**Obtain Sequences**
   - Download protein sequences (FASTA format) from [UniProt](https://www.uniprot.org/help/downloads)
   - In the paper, we use a random subset of UniRef50, but this is large and slow to download so for this walkthrough we'll use Swiss-Prot, which we have found also works for training SAEs.

```bash
# Download sequences
wget -P $INTERPLM_DATA/uniprot/ https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz

# Select random subset and filter to proteins with length < 1022 for ESM-2 compatibility
# Adjust num_proteins to increase the number of proteins kept
python scripts/subset_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/uniprot_sprot.fasta.gz \
    --output_file $INTERPLM_DATA/uniprot/subset.fasta \
    --num_proteins 5000

# Shard fasta into smaller files for training and evaluation
python scripts/shard_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/subset.fasta \
    --output_dir $INTERPLM_DATA/uniprot_shards/ \
    --proteins_per_shard 1000  # 1000/shard -> ~0.5GB/shard for ESM-2 (8M)

# Set aside shard 0 as evaluation data (100 proteins for fidelity testing)
mkdir -p $INTERPLM_DATA/eval_shards/
mv $INTERPLM_DATA/uniprot_shards/shard_0.fasta $INTERPLM_DATA/eval_shards/
```

**Generate protein embeddings for training**
```bash
# Extract embeddings for training shards only (shard_0 is held out for evaluation)
python scripts/extract_embeddings.py \
    --fasta_dir $INTERPLM_DATA/uniprot_shards/ \
    --output_dir $INTERPLM_DATA/training_embeddings/esm2_8m/ \
    --embedder_type esm \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layers $LAYER \
    --batch_size 32
```

> **Note:** The training script automatically uses the first 100 sequences from `eval_shards/shard_0.fasta` for fidelity evaluation.

### 2. Train Sparse Autoencoders

```bash
# Train a Standard ReLU SAE on ESM embeddings (uses $LAYER from Step 1)
python examples/train_basic_sae.py
```

This trains an SAE with 320D embeddings → 1280 features (4x expansion), L1 penalty of 0.06, and 9,500 training steps. The script automatically:
- Trains on embeddings from `training_embeddings/esm2_8m/layer_$LAYER`
- Runs comprehensive evaluation at the end using 100 sequences from shard_0 FASTA
- Saves model to `models/walkthrough_model/layer_$LAYER/ae.pt`
- Saves config and evaluation results (`final_evaluation.yaml`)

**Evaluation metrics** (from `final_evaluation.yaml`):
- **Downstream Task Fidelity**: How well the SAE preserves ESM's masked token prediction (100% = perfect)
- **Reconstruction Quality**: Variance explained and MSE
- **Sparsity**: L0 sparsity, dead features, activation frequency

**Optional - Evaluate on different data:**
```bash
# Evaluate on a different protein set
python scripts/evaluate_sae.py \
    --sae_path models/walkthrough_model/layer_$LAYER/ae.pt \
    --fasta_file $INTERPLM_DATA/uniprot_shards/shard_1.fasta \
    --model_name esm2_t6_8M_UR50D \
    --layer $LAYER \
    --max_proteins 100 \
    --output_file results/custom_eval.yaml
```
> Tip: Use `--skip_fidelity` for ~10x speedup if you only need reconstruction and sparsity metrics.

**To explore different architectures**, see `examples/train_multiple_sae_architectures.py` for examples of Top-K, Jump ReLU, and Batch Top-K SAEs, along with custom hyperparameters, W&B logging, and checkpoint resumption.

### 3. Analyze associations between feature activations and UniProtKB annotations

1. Extract quantitative binary concept labels from UniProtKB data. We provide a curated subset of 1000 Swiss-Prot proteins with dense annotations for the walkthrough. For larger-scale analysis, you can download custom data from UniProt.

**Option A: Use included subset (recommended for walkthrough)**
```bash
# Use the provided curated subset (1000 proteins with dense annotations)
python -m interplm.analysis.concepts.extract_annotations \
    --input_uniprot_path data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz \
    --output_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --n_shards 8 \
    --min_required_instances 10
```

<details>
<summary><b>Option B: Download custom UniProtKB data</b></summary>

For larger-scale analysis or custom protein sets, download data directly from UniProt:

```bash
# Example: Download subset of mouse proteins with structures and high-quality annotations
mkdir -p $INTERPLM_DATA/annotations/uniprotkb
wget -O "${INTERPLM_DATA}/annotations/uniprotkb/proteins.tsv.gz" \
  "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Clength%2Csequence%2Cec%2Cft_act_site%2Cft_binding%2Ccc_cofactor%2Cft_disulfid%2Cft_carbohyd%2Cft_lipid%2Cft_mod_res%2Cft_signal%2Cft_transit%2Cft_helix%2Cft_turn%2Cft_strand%2Cft_coiled%2Ccc_domain%2Cft_compbias%2Cft_domain%2Cft_motif%2Cft_region%2Cft_zn_fing%2Cxref_alphafolddb&format=tsv&query=%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29+AND+%28annotation_score%3A5%29+AND+%28model_organism%3A10090%29+AND+%28length%3A%5B1+TO+400%5D%29"

# Then extract annotations
python -m interplm.analysis.concepts.extract_annotations \
    --input_uniprot_path $INTERPLM_DATA/annotations/uniprotkb/proteins.tsv.gz \
    --output_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --n_shards 8 \
    --min_required_instances 10
```

To download all of Swiss-Prot (used in the paper), remove the query filters from the URL. For larger datasets, increase `--n_shards` and `--min_required_instances` accordingly.
</details>

2. Convert the protein sequences to embeddings
```bash
# Extract embeddings for annotated proteins (uses $LAYER from Step 1)
# These will include boundaries for per-protein analysis
python scripts/embed_annotations.py \
    --input_dir $INTERPLM_DATA/annotations/uniprotkb/processed/ \
    --output_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --embedder_type esm \
    --model_name facebook/esm2_t6_8M_UR50D \
    --layer $LAYER \
    --batch_size 32
```

3. Normalize the SAEs based on the max activating example across a random sample. UniRef50 or any other dataset can be used here for normalization, but we'll default to using the Swiss-Prot data we just embedded.
```bash
python -m interplm.sae.normalize \
    --sae_dir models/walkthrough_model/layer_$LAYER \
    --aa_embds_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER
```
4.  Create evaluation sets with different shards of data. Adjust the numbers here based on the number of shards created in Step 1. This step also filters out any concepts that have do not have many examples in your validation sets.
```bash
# Create validation and test sets (arguments are start and end shard indices, inclusive)
python -m interplm.analysis.concepts.prepare_eval_set \
    --valid_shard_range 0 3 \
    --test_shard_range 4 7 \
    --uniprot_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --min_aa_per_concept 1000 \
    --min_domains_per_concept 25
```

5. Compare all features to all concepts at each threshold

```bash
for EVAL_SET in valid test
do
    # First track classification metrics (tp,fp,etc) on each shard
    python -m interplm.analysis.concepts.compare_activations \
            --sae_dir models/walkthrough_model/layer_$LAYER \
            --aa_embds_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
            --eval_set_dir $INTERPLM_DATA/annotations/uniprotkb/processed/${EVAL_SET}/ \
            --output_dir results/${EVAL_SET}_counts/ && \

    # Then combine all shards to calculate F1 scores
    python -m interplm.analysis.concepts.calculate_f1 \
    --eval_res_dir results/${EVAL_SET}_counts \
    --eval_set_dir $INTERPLM_DATA/annotations/uniprotkb/processed/${EVAL_SET}/
done

# Report metrics on test set based on pairs selected in valid set
python -m interplm.analysis.concepts.report_metrics \
    --valid_path results/valid_counts/concept_f1_scores.csv \
    --test_path results/test_counts/concept_f1_scores.csv
```

### 4. Collect Feature Activations

Find the maximum activating proteins for each feature and compute feature statistics. This step is **required** before creating the dashboard.

```bash
# Collect top activating proteins and compute feature statistics
python scripts/collect_feature_activations.py \
    --sae_dir models/walkthrough_model/layer_$LAYER/ \
    --embeddings_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --metadata_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --shard_range 0 7

# Optional: adjust activation threshold (default 0.05)
# --activation_threshold 0.1  # Only count activations > 0.1 as "activated"
```

This will save to `models/walkthrough_model/layer_$LAYER/`:
- `max_activations_per_feature.pt` - Maximum activation value for each feature
- `Per_feature_statistics.yaml` - Frequency and percentage statistics
  - `Per_prot_frequency_of_any_activation`: % of proteins with any activation
  - `Per_prot_pct_activated_when_present`: Avg % of amino acids activated when feature is present
- `Per_feature_max_examples.yaml` - Top activating proteins per feature
- `Per_feature_quantile_examples.yaml` - Lower quantile examples

### 5. InterPLM Dashboard
The dashboard provides interactive visualization of SAE features and their activations on real protein sequences. You can explore features at [InterPLM.ai](https://interPLM.ai), or create your own dashboard to visualize your trained SAEs.

**Create a dashboard cache:**
```bash
# Create dashboard from the walkthrough SAE (uses $LAYER from Step 1)
python scripts/create_dashboard.py
```

This script will:
- Load the trained SAE and feature analysis results from Step 4
- Create a dashboard cache at `data/dashboard_cache/walkthrough/`
- Automatically include concept enrichment results if available (from Step 3)
- Prepare interactive visualizations

**Launch the dashboard:**
```bash
# Start the dashboard - provide the full path to the cache directory
streamlit run interplm/dashboard/app.py -- --cache_dir $INTERPLM_DATA/dashboard_cache/walkthrough
```

>**Note**: After launching, access the dashboard at http://localhost:8501

The dashboard will show:
- Feature activation patterns across proteins
- Top activating proteins for each feature
- Concept associations (if Step 3 was completed)
- Interactive feature exploration

Now you can scale up the training and concept-evaluation pipelines to explore a broader range of protein language model features. Increasing the training data, adjusting hyperparameters, and expanding the concept evaluation set will help identify features corresponding to other structural motifs, binding sites, and functional domains.


## Extending to Other Protein Models

InterPLM is designed to work with any protein embedding model (language models, structure prediction models, etc.). To add support for a new model:

1. Create a new embedder in `interplm/embedders/` that implements the `BaseEmbedder` interface
2. Use your embedder exactly like ESM in the walkthrough above

See [Adding Your Own Protein Embedder](interplm/embedders/README.md) for detailed instructions.

## Development

### Adding a New SAE Architecture
1. Add the model to `interplm/sae/dictionary.py`
2. Add the trainer to `interplm/train/trainers/your_trainer.py`
3. Import the new trainer in `interplm/train/trainers/__init__.py`

## Citation

If you use InterPLM in your research, please cite:

```bibtex
@article{simon2025interplm,
  title={InterPLM: discovering interpretable features in protein language models via sparse autoencoders},
  author={Simon, Elana and Zou, James},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/s41592-025-02836-7},
  url={https://www.nature.com/articles/s41592-025-02836-7}
}
```

**Preprint version:**
```bibtex
@article{simon2024interplm_preprint,
  title={InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders},
  author={Simon, Elana and Zou, James},
  journal={bioRxiv},
  pages={2024.11.14.623630},
  year={2024},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2024.11.14.623630},
  url={https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1}
}
```

## Contact

- Open an [issue](https://github.com/ElanaPearl/InterPLM/issues) on GitHub
- Email: epsimon [at] stanford [dot] edu
