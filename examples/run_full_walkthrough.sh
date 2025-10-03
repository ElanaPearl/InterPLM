#!/bin/bash
set -e  # Exit on any error

# Full InterPLM Walkthrough Script
# This script runs all steps from the README walkthrough to verify the complete pipeline.
# It uses smaller dataset sizes for faster testing.
#
# IMPORTANT: Run this script with the interplm conda environment activated:
#   conda activate interplm
#   bash examples/run_full_walkthrough.sh

echo "============================================"
echo "InterPLM Full Walkthrough Test"
echo "============================================"
echo ""

# Check conda environment
if ! python -c "import tap" 2>/dev/null; then
    echo "Error: interplm conda environment not activated"
    echo "Please run:"
    echo "  conda activate interplm"
    echo "  bash examples/run_full_walkthrough.sh"
    exit 1
fi

# Check required environment variables
if [ -z "$INTERPLM_DATA" ]; then
    echo "Error: INTERPLM_DATA environment variable not set"
    echo "Please run: export INTERPLM_DATA=/path/to/data"
    exit 1
fi

if [ -z "$LAYER" ]; then
    echo "Warning: LAYER not set, using default LAYER=4"
    export LAYER=4
fi

# ============================================
# HYPERPARAMETERS - Configure walkthrough here
# ============================================

# Training data parameters
NUM_PROTEINS=2000
PROTEINS_PER_SHARD=500

# Protein embedder parameters
EMBEDDER_TYPE="esm"
MODEL_NAME="facebook/esm2_t6_8M_UR50D"
BATCH_SIZE=32

# Swiss-Prot annotation parameters
ANNOTATION_INPUT="data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz"
MIN_REQUIRED_INSTANCES=25
N_SHARDS=6

# Evaluation set parameters
MIN_AA_PER_CONCEPT=1000
MIN_DOMAINS_PER_CONCEPT=25
VALID_SHARD_START=0; VALID_SHARD_END=$((N_SHARDS/2 - 1)); TEST_SHARD_START=$((N_SHARDS/2)); TEST_SHARD_END=$((N_SHARDS - 1))

# Output directories
MODEL_DIR="models/walkthrough_model"
RESULTS_DIR="results/valid_counts"
DASHBOARD_NAME="walkthrough"

echo "Configuration:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  INTERPLM_DATA: $INTERPLM_DATA"
echo "  LAYER: $LAYER"
echo ""
echo "Hyperparameters:"
echo "  NUM_PROTEINS: $NUM_PROTEINS"
echo "  PROTEINS_PER_SHARD: $PROTEINS_PER_SHARD"
echo "  EMBEDDER: $EMBEDDER_TYPE ($MODEL_NAME)"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  N_SHARDS: $N_SHARDS"
echo "  MODEL_DIR: $MODEL_DIR"
echo ""

# Step 0: Setup
echo "Step 0: Setup directories"
mkdir -p $INTERPLM_DATA/{uniprot,uniprot_shards,eval_shards,training_embeddings,analysis_embeddings,annotations}
echo "✓ Directories created"
echo ""

# Step 1a: Download sequences (skip if already exists)
UNIPROT_FILE="$INTERPLM_DATA/uniprot/uniprot_sprot.fasta.gz"
if [ ! -f "$UNIPROT_FILE" ]; then
    echo "Step 1a: Download Swiss-Prot sequences"
    wget -P $INTERPLM_DATA/uniprot/ https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
    echo "✓ Downloaded Swiss-Prot"
else
    echo "Step 1a: Swiss-Prot already downloaded, skipping"
fi
echo ""

# Step 1b: Subset FASTA
echo "Step 1b: Create protein subset ($NUM_PROTEINS proteins for testing)"
python scripts/subset_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/uniprot_sprot.fasta.gz \
    --output_file $INTERPLM_DATA/uniprot/subset.fasta \
    --num_proteins $NUM_PROTEINS
echo "✓ Subset created"
echo ""

# Step 1c: Shard FASTA
echo "Step 1c: Shard FASTA files ($PROTEINS_PER_SHARD proteins/shard)"
python scripts/shard_fasta.py \
    --input_file $INTERPLM_DATA/uniprot/subset.fasta \
    --output_dir $INTERPLM_DATA/uniprot_shards/ \
    --proteins_per_shard $PROTEINS_PER_SHARD
echo "✓ FASTA sharded"
echo ""

# Step 1d: Set aside eval shard
echo "Step 1d: Move shard_0 to eval"
mv $INTERPLM_DATA/uniprot_shards/shard_0.fasta $INTERPLM_DATA/eval_shards/
echo "✓ Eval shard separated"
echo ""

# Step 1e: Extract embeddings for training
echo "Step 1e: Extract training embeddings (layer $LAYER)"
python scripts/extract_embeddings.py \
    --fasta_dir $INTERPLM_DATA/uniprot_shards/ \
    --output_dir $INTERPLM_DATA/training_embeddings/esm2_8m/ \
    --embedder_type $EMBEDDER_TYPE \
    --model_name $MODEL_NAME \
    --layers $LAYER \
    --batch_size $BATCH_SIZE
echo "✓ Training embeddings extracted"
echo ""

# Step 2: Train SAE
echo "Step 2: Train basic SAE (this will take a few minutes)"
python examples/train_basic_sae.py
echo "✓ SAE trained"
echo ""

# Step 3: Concept analysis setup
echo "Step 3: Setup concept analysis"

# Step 3a: Extract annotations (use provided subset for speed)
echo "Step 3a: Extract UniProtKB annotations"
python -m interplm.analysis.concepts.extract_annotations \
    --input_uniprot_path $ANNOTATION_INPUT \
    --output_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --n_shards $N_SHARDS \
    --min_required_instances $MIN_REQUIRED_INSTANCES
echo "✓ Annotations extracted"
echo ""

# Step 3b: Embed annotations
echo "Step 3b: Extract embeddings for annotated proteins"
python scripts/embed_annotations.py \
    --input_dir $INTERPLM_DATA/annotations/uniprotkb/processed/ \
    --output_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --embedder_type $EMBEDDER_TYPE \
    --model_name $MODEL_NAME \
    --layer $LAYER \
    --batch_size $BATCH_SIZE
echo "✓ Annotation embeddings extracted"
echo ""

# Step 3c: Normalize SAE
echo "Step 3c: Normalize SAE"
python -m interplm.sae.normalize \
    --sae_dir $MODEL_DIR/layer_$LAYER \
    --aa_embds_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER
echo "✓ SAE normalized"
echo ""

# Step 3d: Create evaluation sets
echo "Step 3d: Create validation and test sets"
python -m interplm.analysis.concepts.prepare_eval_set \
    --valid_shard_range $VALID_SHARD_START $VALID_SHARD_END \
    --test_shard_range $TEST_SHARD_START $TEST_SHARD_END \
    --uniprot_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --min_aa_per_concept $MIN_AA_PER_CONCEPT \
    --min_domains_per_concept $MIN_DOMAINS_PER_CONCEPT
echo "✓ Eval sets created"
echo ""

# Step 3e: Compare activations (only on valid set for speed)
echo "Step 3e: Compare feature activations to concepts (validation set only)"
python -m interplm.analysis.concepts.compare_activations \
    --sae_dir $MODEL_DIR/layer_$LAYER \
    --aa_embds_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --eval_set_dir $INTERPLM_DATA/annotations/uniprotkb/processed/valid/ \
    --output_dir $RESULTS_DIR/
echo "✓ Activations compared"
echo ""

# Step 3f: Calculate F1 scores
echo "Step 3f: Calculate F1 scores"
python -m interplm.analysis.concepts.calculate_f1 \
    --eval_res_dir $RESULTS_DIR \
    --eval_set_dir $INTERPLM_DATA/annotations/uniprotkb/processed/valid/
echo "✓ F1 scores calculated"
echo ""

# Step 4: Collect feature activations
echo "Step 4: Collect feature activations for dashboard"
python scripts/collect_feature_activations.py \
    --sae_dir $MODEL_DIR/layer_$LAYER/ \
    --embeddings_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --metadata_dir $INTERPLM_DATA/annotations/uniprotkb/processed \
    --shard_range $VALID_SHARD_START $TEST_SHARD_END
echo "✓ Feature activations collected"
echo ""

# Step 5: Create dashboard
echo "Step 5: Create dashboard cache"
python scripts/create_dashboard.py \
    --sae_path $MODEL_DIR/layer_$LAYER/ae.pt \
    --embeddings_dir $INTERPLM_DATA/analysis_embeddings/esm2_8m/layer_$LAYER \
    --layer $LAYER \
    --metadata_path $ANNOTATION_INPUT \
    --dashboard_name $DASHBOARD_NAME \
    --model_name $EMBEDDER_TYPE \
    --model_type $MODEL_NAME \
    --shard_range $VALID_SHARD_START $TEST_SHARD_END \
    --concept_enrichment_path $RESULTS_DIR/concept_f1_scores.csv
echo "✓ Dashboard created"
echo ""

echo "============================================"
echo "✅ Full Walkthrough Complete!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  - Trained SAE: $MODEL_DIR/layer_$LAYER/ae.pt"
echo "  - Evaluation: $INTERPLM_DATA/evaluation_results.yaml"
echo "  - Concept F1: $RESULTS_DIR/concept_f1_scores.csv"
echo "  - Dashboard cache: $INTERPLM_DATA/dashboard_cache/$DASHBOARD_NAME/"
echo ""
echo "To view the dashboard, run:"
echo "  streamlit run interplm/dashboard/app.py -- --cache_dir $INTERPLM_DATA/dashboard_cache/$DASHBOARD_NAME"
echo ""
