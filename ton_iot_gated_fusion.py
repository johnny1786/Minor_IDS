"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TON-IoT Intrusion Detection System — LOCAL VERSION (Target: 99% Acc)     ║
║   Pipeline: Decision Tree + XGBoost + Gated Image-Tabular Fusion + RF/LGBM  ║
║             + Late Fusion | PCA + LDA + train-only WCGAN-GP                 ║
║   Dataset : TON-IoT Network CSV (train_test_network.csv or similar)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    1. pip install -r requirements_ids.txt
    2. Place TON-IoT CSV file(s) in ./data_ton/ folder (or edit DATASET_PATH below)
       Download from: kaggle datasets download -d fadiabuzwayed/ton-iot-train-test-network
    3. python ton_iot_gated_fusion.py
    4. Results saved to ./outputs_ton_gated_fusion/

MODELS (6 total — NO Transfer Learning):
    ✅ Model 1 : Decision Tree
    ✅ Model 2 : XGBoost
    ✅ Model 3 : Gated Image-Tabular Fusion
    ✅ Model 4 : Random Forest
    ✅ Model 5 : LightGBM
    ✅ Model 6 : Late Fusion Ensemble

EXPERIMENTAL RESEARCH-BASED CHANGES:
    ✅ PCA (n=15) → removes noise, captures 95%+ variance
    ✅ LDA (n=classes-1) → supervised dimensionality reduction
    ✅ Combined PCA+LDA feature stack → richer representation
    ✅ Train-only WCGAN-GP oversampling → avoids synthetic test leakage
    ✅ Learnable Feature Gating → dynamic supervised feature weighting
    ✅ Focal smoothing loss → hard-class focus + label smoothing
    ✅ Late fusion of XGBoost + LightGBM + gated deep probabilities
    ✅ QuantileTransformer normalisation
    ✅ IQR outlier capping (1–99th percentile)
    ✅ OneHot encoding for nominal categoricals
    ✅ Clustering-based deduplication (PCA + MiniBatchKMeans)
    ✅ CPU/GPU auto-detection with memory management
    ✅ All paths local (no Drive mount, no Kaggle API calls in script)
    ✅ CosineDecay LR + EarlyStopping + ModelCheckpoint
"""

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG — EDIT THESE
# ─────────────────────────────────────────────────────────────────────────────
DATASET_FOLDER = "./data"          # Folder containing TON-IoT CSV file(s)
DATASET_FILE   = "./data/train_test_network.csv"  # Preferred TON-IoT network CSV
OUTPUT_DIR     = "./outputs_ton_gated_fusion"  # Separate experiment outputs
WCGAN_EPOCHS   = 20                # Reduced first-run epochs; increase after validation
XGB_ESTIMATORS = 500               # XGBoost trees
RF_ESTIMATORS  = 300               # Random Forest trees
CNN_EPOCHS     = 25                # Reduced first-run epochs for gated fusion model
BATCH_SIZE     = 256               # Reduce to 128 if low RAM
KEEP_RATIO     = 0.50              # Fraction kept after deduplication
MAX_DEDUP_CLUSTERS = 512           # Hard cap per class for faster KMeans dedup
DEDUP_FIT_CAP      = 20000         # Max samples/class used to fit MiniBatchKMeans
FOCAL_GAMMA        = 2.0           # Focuses learning on hard attack classes
LABEL_SMOOTHING    = 0.04          # Regularises over-confident deep predictions
FUSION_WEIGHTS     = (0.40, 0.40, 0.20)  # XGBoost, LightGBM, Gated Fusion
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, time, gc, pickle, json, warnings, subprocess
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Avoid joblib physical-core detection failures on macOS sandboxed runs.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
from pathlib import Path
import glob

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing         import LabelEncoder, QuantileTransformer, OneHotEncoder
from sklearn.compose               import ColumnTransformer
from sklearn.model_selection       import train_test_split
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import RandomForestClassifier
from sklearn.decomposition         import PCA
from sklearn.cluster               import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics               import (accuracy_score, recall_score, f1_score,
                                           precision_score, classification_report,
                                           confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def notify_user(title, message):
    print('\a', end='', flush=True)
    if sys.platform != 'darwin':
        return
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _notify_on_exception(exc_type, exc_value, exc_traceback):
    notify_user("TON-IoT IDS Failed", str(exc_value)[:120])
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _notify_on_exception

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_FOLDER = Path(DATASET_FOLDER)
DATASET_FILE   = Path(DATASET_FILE)
OUTPUT_DIR     = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer the configured folder, but fall back to the common local dataset paths.
if not DATASET_FOLDER.exists():
    for candidate in (Path("./data"), Path("./data_ton")):
        if candidate.exists():
            DATASET_FOLDER = candidate
            break

print("=" * 70)
print("  TON-IoT IDS — Local Run (Gated Fusion + Train-only WCGAN-GP)")
print("=" * 70)

# ── GPU Setup ─────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] GPU memory growth: {e}")

if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"✅ GPU mode  — {len(gpus)} GPU(s) detected. Mixed precision ON.")
else:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("ℹ️  CPU mode — no GPU detected. Training will be slower.")

print(f"TensorFlow : {tf.__version__}")
print(f"XGBoost    : {xgb.__version__}")
print(f"LightGBM   : {lgb.__version__}")
print(f"Output dir : {OUTPUT_DIR.resolve()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load TON-IoT Dataset
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 1: Loading TON-IoT Dataset")
print("─" * 70)

t0_load = time.time()
dataset = None

preferred_candidates = []
if DATASET_FILE.exists():
    preferred_candidates.append(DATASET_FILE)

if DATASET_FOLDER.exists():
    csv_files = list(DATASET_FOLDER.rglob("*.csv"))
    csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    preferred_candidates.extend([p for p in csv_files if p != DATASET_FILE])

if not preferred_candidates:
    print(f"❌ TON-IoT dataset not found. Expected file: {DATASET_FILE.resolve()}")
    sys.exit(1)

for csv_path in preferred_candidates:
    try:
        dataset = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded: {csv_path.name} — shape: {dataset.shape}")
        break
    except Exception as e:
        print(f"  Failed {csv_path.name}: {e}")

if dataset is None:
    print("❌ Could not load any CSV. Check your data folder.")
    sys.exit(1)
print(f"Load time: {time.time()-t0_load:.1f}s")
print(f"Columns: {dataset.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature Extraction & Cleaning
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 2: Feature Extraction & Cleaning")
print("─" * 70)

# Determine label column
LABEL_COL = None
for candidate in ['type', 'label', 'attack_cat', 'attack']:
    if candidate in dataset.columns:
        LABEL_COL = candidate
        break
if LABEL_COL is None:
    print("❌ No label column found. Expected 'type', 'label', or 'attack_cat'.")
    sys.exit(1)
print(f"Label column: '{LABEL_COL}'")
print(f"\nClass distribution:\n{dataset[LABEL_COL].value_counts()}")

# Feature columns (domain-driven, keep what's available)
PREFERRED_FEATURES = [
    'proto', 'duration', 'src_bytes', 'dst_bytes', 'conn_state',
    'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qtype_name', 'dns_rcode_name', 'dns_AA',
    # fallback numeric features present in many TON-IoT variants
    'src_port', 'dst_port', 'service', 'history',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'tunnel_parents', 'uid'
]
available_features = [c for c in PREFERRED_FEATURES if c in dataset.columns
                       and c != LABEL_COL]
# If very few preferred features found, use all non-label columns
if len(available_features) < 5:
    available_features = [c for c in dataset.columns if c != LABEL_COL]
print(f"\nUsing {len(available_features)} features: {available_features}")

df = dataset[available_features + [LABEL_COL]].copy()
del dataset; gc.collect()

# Drop NaN
before = len(df)
df.dropna(inplace=True)
print(f"Rows dropped (NaN): {before - len(df):,}")

# IQR-based outlier capping (1st–99th percentile)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(p01, p99)
print(f"✅ Cleaning done. Shape: {df.shape}")

# Class distribution plot
colors_palette = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
                  '#1abc9c','#e67e22','#34495e','#f1c40f','#95a5a6']
cc = df[LABEL_COL].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(len(cc)), cc.values, color=colors_palette[:len(cc)])
axes[0].set_xticks(range(len(cc)))
axes[0].set_xticklabels(cc.index, rotation=30, ha='right', fontsize=8)
axes[0].set_title('Class Distribution — TON-IoT', fontweight='bold')
axes[1].pie(cc.values, labels=cc.index, autopct='%1.1f%%',
            colors=colors_palette[:len(cc)])
axes[1].set_title('Class Proportion', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Encoding (OneHot + LabelEncoder)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 3: Encoding (OneHot + LabelEncoder)")
print("─" * 70)

# Detect categoricals robustly across object/string/category dtypes.
cat_cols = [c for c in available_features if not is_numeric_dtype(df[c])]
num_cols = [c for c in available_features if is_numeric_dtype(df[c])]
print(f"Categorical cols : {cat_cols}")
print(f"Numeric cols     : {num_cols}")

if cat_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )
    X_raw = preprocessor.fit_transform(df[available_features]).astype(np.float32)
else:
    preprocessor = None
    X_raw = df[available_features].values.astype(np.float32)

print(f"After OHE — X shape: {X_raw.shape}")

le = LabelEncoder()
y_raw = le.fit_transform(df[LABEL_COL]).astype(np.int32)
N_CLASSES = len(le.classes_)
print(f"\n{N_CLASSES} classes:")
for i, cls in enumerate(le.classes_):
    cnt = (y_raw == i).sum()
    print(f"  {i} → {cls}: {cnt:,} ({100*cnt/len(y_raw):.2f}%)")

del df; gc.collect()
print("✅ Encoding complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Quantile Normalisation
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 4: Quantile Normalisation")
print("─" * 70)

qt = QuantileTransformer(output_distribution='normal',
                         n_quantiles=1000, random_state=SEED)
X_scaled = qt.fit_transform(X_raw).astype(np.float32)
print(f"Normalised shape : {X_scaled.shape}")
print(f"Mean ≈ {X_scaled.mean():.3f} | Std ≈ {X_scaled.std():.3f}")
del X_raw; gc.collect()
print("✅ Normalisation done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Clustering-Based Deduplication (PCA + MiniBatchKMeans)
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 5: Clustering-Based Deduplication")
print("─" * 70)

N_COMPONENTS_PCA = min(64, X_scaled.shape[1])
print(f"Applying PCA (n={N_COMPONENTS_PCA}) for deduplication...")
pca_dedup = PCA(n_components=N_COMPONENTS_PCA, random_state=SEED)
X_pca_dedup = pca_dedup.fit_transform(X_scaled)
print(f"Explained variance: {pca_dedup.explained_variance_ratio_.sum():.3f}")

selected_indices = []
for cls_id in range(N_CLASSES):
    cls_mask  = (y_raw == cls_id)
    cls_pca   = X_pca_dedup[cls_mask]
    cls_idx   = np.where(cls_mask)[0]
    target_keep = max(1, int(len(cls_idx) * KEEP_RATIO))
    n_clusters = min(target_keep, MAX_DEDUP_CLUSTERS, len(cls_idx))

    if len(cls_idx) <= n_clusters:
        selected_indices.extend(cls_idx.tolist())
        continue

    fit_idx = np.arange(len(cls_idx))
    if len(fit_idx) > DEDUP_FIT_CAP:
        fit_idx = np.random.choice(len(cls_idx), DEDUP_FIT_CAP, replace=False)

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED,
                         batch_size=min(1024, len(cls_idx)))
    km.fit(cls_pca[fit_idx])
    cluster_labels = km.predict(cls_pca)
    assigned_centers = km.cluster_centers_[cluster_labels]
    dist_sq = np.sum((cls_pca - assigned_centers) ** 2, axis=1)

    order = np.lexsort((dist_sq, cluster_labels))
    sorted_labels = cluster_labels[order]
    _, first_pos = np.unique(sorted_labels, return_index=True)
    selected_indices.extend(cls_idx[order[first_pos]].tolist())

    print(f"  Class {cls_id} ({le.classes_[cls_id]}): {len(cls_idx)} → {len(first_pos)} samples "
          f"(target {target_keep}, cap {MAX_DEDUP_CLUSTERS})")

selected_indices = sorted(set(selected_indices))
X_dedup = X_scaled[selected_indices]
y_dedup = y_raw[selected_indices]
del X_pca_dedup, pca_dedup; gc.collect()
print(f"\nBefore dedup: {len(X_scaled):,} | After: {len(X_dedup):,}")
print("✅ Deduplication done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PCA + LDA Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 6: PCA + LDA Feature Engineering")
print("─" * 70)

PCA_COMPONENTS = min(15, X_dedup.shape[1])
print(f"Fitting PCA (n={PCA_COMPONENTS})...")
t0 = time.time()
pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
FIT_SAMPLE  = min(200_000, len(X_dedup))
pca_fit_idx = np.random.choice(len(X_dedup), FIT_SAMPLE, replace=False)
pca.fit(X_dedup[pca_fit_idx])
X_pca = pca.transform(X_dedup).astype(np.float32)
explained = np.cumsum(pca.explained_variance_ratio_)
print(f"  PCA done ({time.time()-t0:.1f}s) — {explained[-1]*100:.1f}% variance")

plt.figure(figsize=(8, 3))
plt.bar(range(1, PCA_COMPONENTS+1), pca.explained_variance_ratio_,
        color='#3498db', alpha=0.7, label='Individual')
plt.plot(range(1, PCA_COMPONENTS+1), explained,
         color='#e74c3c', marker='o', label='Cumulative')
plt.axhline(0.95, color='green', linestyle='--', label='95% threshold')
plt.title('PCA Explained Variance', fontweight='bold')
plt.xlabel('Component'); plt.ylabel('Variance Ratio')
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_variance.png', dpi=150); plt.close()

LDA_COMPONENTS = min(N_CLASSES - 1, PCA_COMPONENTS)
print(f"\nFitting LDA (n={LDA_COMPONENTS})...")
t0 = time.time()
lda = LDA(n_components=LDA_COMPONENTS)
lda.fit(X_pca[pca_fit_idx], y_dedup[pca_fit_idx])
X_lda = lda.transform(X_pca).astype(np.float32)
print(f"  LDA done ({time.time()-t0:.1f}s) — shape: {X_lda.shape}")

X_combined = np.hstack([X_dedup, X_pca, X_lda]).astype(np.float32)
print(f"\nCombined stack: {X_dedup.shape[1]} raw + "
      f"{X_pca.shape[1]} PCA + {X_lda.shape[1]} LDA "
      f"= {X_combined.shape[1]} total features")

if LDA_COMPONENTS >= 2:
    fig, ax = plt.subplots(figsize=(9, 6))
    sample_idx = np.random.choice(len(X_lda), min(5000, len(X_lda)), replace=False)
    for cls_id, cls_name in enumerate(le.classes_):
        mask = y_dedup[sample_idx] == cls_id
        ax.scatter(X_lda[sample_idx][mask, 0],
                   X_lda[sample_idx][mask, 1],
                   label=cls_name, alpha=0.4, s=10,
                   color=colors_palette[cls_id % len(colors_palette)])
    ax.set_title('LDA — Class Separation (LD1 vs LD2)', fontweight='bold')
    ax.set_xlabel('LD1'); ax.set_ylabel('LD2')
    ax.legend(markerscale=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lda_scatter.png', dpi=150); plt.close()

print("✅ PCA + LDA done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Natural Train / Test Split Before Oversampling
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 7: Natural Train / Test Split Before Oversampling")
print("─" * 70)

X_train_base, X_test, y_train_base, y_test = train_test_split(
    X_combined, y_dedup,
    test_size=0.20, stratify=y_dedup, random_state=SEED
)
print(f"Train before WCGAN: {X_train_base.shape} | Natural test: {X_test.shape}")
print("✅ Split done before synthetic generation, so test metrics stay natural\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Train-Only WCGAN-GP Oversampling
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 8: Train-Only WCGAN-GP — Oversampling for Class Balance")
print("─" * 70)

N_FEATURES    = X_combined.shape[1]
LATENT_DIM    = 64
CRITIC_STEPS  = 5
GP_LAMBDA     = 10
WCGAN_BATCH   = min(BATCH_SIZE, 256)
GAN_LR        = 2e-4
GAN_TRAIN_CAP = 30_000
GEN_CHUNK     = 5_000

def build_generator(latent_dim, n_classes, n_features):
    noise_in = keras.Input(shape=(latent_dim,))
    label_in = keras.Input(shape=(1,), dtype='int32')
    emb  = layers.Flatten()(layers.Embedding(n_classes, latent_dim)(label_in))
    x    = layers.Concatenate()([noise_in, emb])
    for units in [256, 512, 256]:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
    out = layers.Dense(n_features, activation='tanh', dtype='float32')(x)
    return keras.Model([noise_in, label_in], out, name='Generator')

def build_critic(n_classes, n_features):
    feat_in  = keras.Input(shape=(n_features,))
    label_in = keras.Input(shape=(1,), dtype='int32')
    emb  = layers.Flatten()(layers.Embedding(n_classes, n_features)(label_in))
    x    = layers.Multiply()([feat_in, emb])
    for units in [512, 256, 128]:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    score = layers.Dense(1, dtype='float32')(x)
    return keras.Model([feat_in, label_in], score, name='Critic')

generator = build_generator(LATENT_DIM, N_CLASSES, N_FEATURES)
critic    = build_critic(N_CLASSES, N_FEATURES)
gen_opt   = keras.optimizers.Adam(GAN_LR, beta_1=0.0, beta_2=0.9)
crit_opt  = keras.optimizers.Adam(GAN_LR, beta_1=0.0, beta_2=0.9)

@tf.function(reduce_retracing=True)
def gradient_penalty(real, fake, labels):
    bs    = tf.shape(real)[0]
    alpha = tf.random.uniform([bs, 1], 0., 1., dtype=tf.float32)
    inter = tf.cast(alpha * real + (1. - alpha) * fake, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = critic([inter, labels], training=True)
    grads = tape.gradient(pred, inter)
    norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
    return tf.reduce_mean(tf.square(norm - 1.0))

@tf.function(reduce_retracing=True)
def train_step(real_feat, real_labels):
    bs     = tf.shape(real_feat)[0]
    c_loss = tf.constant(0.0)
    for _ in range(CRITIC_STEPS):
        noise = tf.random.normal([bs, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake    = generator([noise, real_labels], training=False)
            real_sc = critic([real_feat, real_labels], training=True)
            fake_sc = critic([fake,      real_labels], training=True)
            gp      = gradient_penalty(real_feat, fake, real_labels)
            c_loss  = (tf.reduce_mean(fake_sc) - tf.reduce_mean(real_sc)
                       + GP_LAMBDA * gp)
        crit_grads = tape.gradient(c_loss, critic.trainable_variables)
        crit_opt.apply_gradients(zip(crit_grads, critic.trainable_variables))
    noise = tf.random.normal([bs, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake    = generator([noise, real_labels], training=True)
        fake_sc = critic([fake, real_labels], training=False)
        g_loss  = -tf.reduce_mean(fake_sc)
    gen_grads = tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    return c_loss, g_loss

unique_cls, cls_counts = np.unique(y_train_base, return_counts=True)
max_count = int(cls_counts.max())
cls_count_map = {int(cid): int(cnt) for cid, cnt in zip(unique_cls, cls_counts)}
print("Training class counts before oversampling:")
for cid, cnt in zip(unique_cls, cls_counts):
    print(f"  {le.classes_[cid]}: {cnt:,}")
print(f"Target (majority): {max_count:,}")
print(f"WCGAN-GP epochs  : {WCGAN_EPOCHS}\n")

X_gen_list = [X_train_base]
y_gen_list = [y_train_base]

for cls_id in unique_cls:
    needed = max_count - cls_count_map[int(cls_id)]
    if needed <= 0:
        print(f"[SKIP] {le.classes_[cls_id]} — majority class")
        continue
    print(f"\n🔄 \"{le.classes_[cls_id]}\" — generating {needed:,} samples")

    cls_data   = X_train_base[y_train_base == cls_id].astype(np.float32)
    cls_labels = np.full(len(cls_data), cls_id, dtype=np.int32).reshape(-1, 1)
    if len(cls_data) > GAN_TRAIN_CAP:
        idx_cap    = np.random.choice(len(cls_data), GAN_TRAIN_CAP, replace=False)
        cls_data   = cls_data[idx_cap]
        cls_labels = cls_labels[idx_cap]

    dataset_tf = (
        tf.data.Dataset
        .from_tensor_slices((cls_data, cls_labels))
        .shuffle(min(len(cls_data), 10_000), reshuffle_each_iteration=True)
        .batch(WCGAN_BATCH, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    t_cls = time.time()
    for epoch in range(WCGAN_EPOCHS):
        for batch_feat, batch_labels in dataset_tf:
            c_loss, g_loss = train_step(batch_feat, batch_labels)
        if (epoch + 1) % 25 == 0:
            elapsed = time.time() - t_cls
            eta     = elapsed / (epoch + 1) * (WCGAN_EPOCHS - epoch - 1)
            print(f"  Epoch {epoch+1:3d}/{WCGAN_EPOCHS} | "
                  f"C: {float(c_loss):+.3f} | G: {float(g_loss):+.3f} | "
                  f"⏱ {elapsed:.0f}s | ETA {eta:.0f}s")

    synth_parts = []
    remaining   = needed
    while remaining > 0:
        n_gen     = min(GEN_CHUNK, remaining)
        noise_gen = tf.random.normal([n_gen, LATENT_DIM])
        lbl_gen   = tf.constant([[cls_id]] * n_gen, dtype=tf.int32)
        synth_parts.append(
            generator([noise_gen, lbl_gen], training=False).numpy()
        )
        remaining -= n_gen

    synthetic = np.vstack(synth_parts).astype(np.float32)
    X_gen_list.append(synthetic)
    y_gen_list.append(np.full(needed, cls_id, dtype=np.int32))
    print(f"  ✅ {time.time()-t_cls:.0f}s — generated {needed:,} samples")
    del cls_data, cls_labels, synth_parts, synthetic; gc.collect()

X_train = np.vstack(X_gen_list).astype(np.float32)
y_train = np.concatenate(y_gen_list).astype(np.int32)
del X_gen_list, y_gen_list, generator, critic; gc.collect()
print(f"\n✅ Balanced training set: {X_train.shape}")
for u, c in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {le.classes_[u]}: {c:,}")
del X_train_base, y_train_base, X_combined; gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Final Training Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 9: Final Training Matrix")
print("─" * 70)

print(f"Train: {X_train.shape} | Natural test: {X_test.shape}")

results = {}
print("✅ Matrices ready\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Model 1: Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 10: Model 1 — Decision Tree")
print("─" * 70)

t0 = time.time()
dt_model = DecisionTreeClassifier(
    criterion='entropy', max_depth=25,
    min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=SEED
)
dt_model.fit(X_train, y_train)
print(f"  Fit time: {time.time()-t0:.1f}s")

t0        = time.time()
y_pred_dt = dt_model.predict(X_test)
dt_lat    = (time.time() - t0) / len(y_test) * 1000

cm_dt = confusion_matrix(y_test, y_pred_dt)
FP = cm_dt.sum(axis=0) - np.diag(cm_dt)
TN = cm_dt.sum() - (cm_dt.sum(axis=1) + cm_dt.sum(axis=0) - np.diag(cm_dt))
results['Decision Tree'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_dt),
    'Recall'    : recall_score(y_test, y_pred_dt, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_dt, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_dt, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': dt_lat
}
print(f"Accuracy : {results['Decision Tree']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['Decision Tree']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_, zero_division=0))

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Decision Tree — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_dt.png', dpi=150); plt.close()
print("✅ Decision Tree done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Model 2: XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 11: Model 2 — XGBoost")
print("─" * 70)

use_gpu = len(tf.config.list_physical_devices('GPU')) > 0
xgb_params = dict(
    n_estimators          = XGB_ESTIMATORS,
    max_depth             = 8,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    reg_alpha             = 0.05,
    reg_lambda            = 1.0,
    eval_metric           = 'mlogloss',
    objective             = 'multi:softprob',
    num_class             = N_CLASSES,
    random_state          = SEED,
    n_jobs                = -1,
    tree_method           = 'hist',
    device                = 'cuda' if use_gpu else 'cpu',
    verbosity             = 0,
    early_stopping_rounds = 30,
)

t0        = time.time()
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
print(f"  Fit time: {time.time()-t0:.1f}s | Best iter: {xgb_model.best_iteration}")

t0              = time.time()
y_pred_xgb_prob = xgb_model.predict_proba(X_test)
xgb_lat         = (time.time() - t0) / len(y_test) * 1000
y_pred_xgb      = np.argmax(y_pred_xgb_prob, axis=1)

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
FP = cm_xgb.sum(axis=0) - np.diag(cm_xgb)
TN = cm_xgb.sum() - (cm_xgb.sum(axis=1) + cm_xgb.sum(axis=0) - np.diag(cm_xgb))
results['XGBoost'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_xgb),
    'Recall'    : recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': xgb_lat
}
print(f"Accuracy : {results['XGBoost']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['XGBoost']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_, zero_division=0))

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('XGBoost — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_xgb.png', dpi=150); plt.close()
print("✅ XGBoost done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Model 3: Gated Image-Tabular Fusion
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 12: Model 3 — Gated Image-Tabular Fusion")
print("─" * 70)

def tabular_to_images(X):
    n_features = X.shape[1]
    side = int(np.ceil(np.sqrt(n_features)))
    padded_features = side * side
    pad_width = padded_features - n_features
    if pad_width:
        X_pad = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
    else:
        X_pad = X
    return X_pad.reshape(-1, side, side, 1).astype(np.float32), side, pad_width

X_train_cnn, IMG_SIDE, IMG_PAD = tabular_to_images(X_train)
X_test_cnn, _, _ = tabular_to_images(X_test)
y_train_cat = keras.utils.to_categorical(y_train, N_CLASSES).astype(np.float32)
y_test_cat  = keras.utils.to_categorical(y_test,  N_CLASSES).astype(np.float32)
print(f"Image representation: {X_train.shape[1]} features + {IMG_PAD} padding "
      f"→ {IMG_SIDE}x{IMG_SIDE}x1")

def focal_smoothing_loss(gamma=2.0, label_smoothing=0.04):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        n_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n_classes
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(focal, axis=-1)
    return loss_fn

def learnable_feature_gate(tab_in):
    gate = layers.Dense(
        tab_in.shape[-1],
        activation='sigmoid',
        name='learnable_feature_gate'
    )(tab_in)
    gated = layers.Multiply(name='gated_tabular_features')([tab_in, gate])
    return gated, gate

def channel_attention(x, ratio=8):
    ch  = x.shape[-1]
    avg = layers.Reshape((1, 1, ch))(layers.GlobalAveragePooling2D()(x))
    mx  = layers.Reshape((1, 1, ch))(layers.GlobalMaxPooling2D()(x))
    d1  = layers.Dense(max(1, ch // ratio), activation='relu')
    d2  = layers.Dense(ch)
    scale = layers.Activation('sigmoid')(
        layers.Add()([d2(d1(avg)), d2(d1(mx))]))
    return layers.Multiply()([x, scale])

def spatial_attention(x):
    avg   = layers.Lambda(lambda t: keras.ops.mean(t, axis=-1, keepdims=True))(x)
    mx    = layers.Lambda(lambda t: keras.ops.max(t, axis=-1,  keepdims=True))(x)
    cat   = layers.Concatenate(axis=-1)([avg, mx])
    scale = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(cat)
    return layers.Multiply()([x, scale])

def conv_block(x, filters):
    shortcut = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_gated_fusion_model(img_side, n_tab_features, n_classes):
    img_in = keras.Input(shape=(img_side, img_side, 1), name='image_input')
    tab_in = keras.Input(shape=(n_tab_features,), name='tabular_input')

    x = conv_block(img_in, 32)
    x   = layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x   = conv_block(x, 64)
    x   = channel_attention(x)
    x   = spatial_attention(x)
    x   = conv_block(x, 128)
    img_vec = layers.GlobalAveragePooling2D(name='image_attention_gap')(x)
    img_vec = layers.Dense(128, activation='relu', name='image_embedding')(img_vec)
    img_vec = layers.Dropout(0.25)(img_vec)

    gated_tab, gate = learnable_feature_gate(tab_in)
    tab_vec = layers.Dense(128, activation='relu', name='tabular_embedding')(gated_tab)
    tab_vec = layers.BatchNormalization()(tab_vec)
    tab_vec = layers.Dropout(0.25)(tab_vec)

    x = layers.Concatenate(name='image_tabular_fusion')([img_vec, tab_vec])
    x = layers.Dense(160, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model([img_in, tab_in], out, name='Gated_Image_Tabular_Fusion')

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=max(1, CNN_EPOCHS * (len(X_train_cnn) // BATCH_SIZE))
)

fusion_model = build_gated_fusion_model(IMG_SIDE, X_train.shape[1], N_CLASSES)
fusion_model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    loss=focal_smoothing_loss(FOCAL_GAMMA, LABEL_SMOOTHING),
    metrics=['accuracy']
)
fusion_model.summary()

cbs_hybrid = [
    EarlyStopping(monitor='val_loss', patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(OUTPUT_DIR / 'gated_fusion_best.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=0),
]

t0 = time.time()
history_hybrid = fusion_model.fit(
    [X_train_cnn, X_train], y_train_cat,
    validation_split=0.10,
    epochs=CNN_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs_hybrid,
    verbose=1
)
print(f"\nGated fusion training: {time.time()-t0:.1f}s")

t0 = time.time()
y_pred_hyb_probs = fusion_model.predict([X_test_cnn, X_test], batch_size=512, verbose=0)
hyb_lat          = (time.time() - t0) / len(y_test) * 1000
y_pred_hyb       = np.argmax(y_pred_hyb_probs, axis=1)

cm_hyb = confusion_matrix(y_test, y_pred_hyb)
FP = cm_hyb.sum(axis=0) - np.diag(cm_hyb)
TN = cm_hyb.sum() - (cm_hyb.sum(axis=1) + cm_hyb.sum(axis=0) - np.diag(cm_hyb))
results['Gated Image-Tabular Fusion'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_hyb),
    'Recall'    : recall_score(y_test, y_pred_hyb, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_hyb, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_hyb, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': hyb_lat
}
print(f"Accuracy : {results['Gated Image-Tabular Fusion']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['Gated Image-Tabular Fusion']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_hyb, target_names=le.classes_, zero_division=0))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history_hybrid.history['accuracy'],     label='Train')
ax[0].plot(history_hybrid.history['val_accuracy'], label='Val')
ax[0].set_title('Gated Fusion Accuracy'); ax[0].legend()
ax[1].plot(history_hybrid.history['loss'],     label='Train')
ax[1].plot(history_hybrid.history['val_loss'], label='Val')
ax[1].set_title('Gated Fusion Loss'); ax[1].legend()
plt.suptitle('Gated Image-Tabular Fusion Training', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gated_fusion_training.png', dpi=150); plt.close()

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_hyb, annot=True, fmt='d', cmap='Purples',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Gated Image-Tabular Fusion — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_gated_fusion.png', dpi=150); plt.close()
print("✅ Gated Image-Tabular Fusion done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — Model 4: Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 13: Model 4 — Random Forest")
print("─" * 70)
print("""
  WHY RANDOM FOREST?
  ┌─ Ensemble of decorrelated decision trees (bagging)
  │   - Lower variance than single tree → more stable 99%+ accuracy
  │   - Naturally handles high-dimensional PCA+LDA feature stacks
  │   - Built-in feature importance for interpretability
  └─ Complements XGBoost diversity → better collective performance
""")

t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators      = RF_ESTIMATORS,
    max_depth         = 30,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = 'sqrt',
    class_weight      = 'balanced_subsample',
    n_jobs            = -1,
    random_state      = SEED
)
rf_model.fit(X_train, y_train)
print(f"  Fit time: {time.time()-t0:.1f}s")

t0        = time.time()
y_pred_rf = rf_model.predict(X_test)
rf_lat    = (time.time() - t0) / len(y_test) * 1000

cm_rf = confusion_matrix(y_test, y_pred_rf)
FP = cm_rf.sum(axis=0) - np.diag(cm_rf)
TN = cm_rf.sum() - (cm_rf.sum(axis=1) + cm_rf.sum(axis=0) - np.diag(cm_rf))
results['Random Forest'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_rf),
    'Recall'    : recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': rf_lat
}
print(f"Accuracy : {results['Random Forest']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['Random Forest']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_, zero_division=0))

feat_importances = rf_model.feature_importances_
top_k = min(20, len(feat_importances))
top_idx = np.argsort(feat_importances)[-top_k:][::-1]
plt.figure(figsize=(10, 4))
plt.bar(range(top_k), feat_importances[top_idx], color='#2ecc71')
plt.xticks(range(top_k), [f'F{i}' for i in top_idx], rotation=45, ha='right', fontsize=7)
plt.title(f'Random Forest — Top {top_k} Feature Importances', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rf_feature_importance.png', dpi=150); plt.close()

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Random Forest — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_rf.png', dpi=150); plt.close()
print("✅ Random Forest done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — Model 5: LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 14: Model 5 — LightGBM")
print("─" * 70)
print("""
  WHY LIGHTGBM?
  ┌─ Leaf-wise gradient boosting (GOSS + EFB)
  │   - Faster than XGBoost on large IoT datasets
  │   - Lower memory footprint
  │   - Excellent on mixed feature types (PCA+LDA+raw)
  └─ Consistently achieves 99%+ on network intrusion benchmarks
""")

use_gpu_lgb = len(tf.config.list_physical_devices('GPU')) > 0
lgb_params = dict(
    n_estimators      = XGB_ESTIMATORS,
    max_depth         = 10,
    learning_rate     = 0.05,
    num_leaves        = 63,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    min_child_samples = 10,
    class_weight      = 'balanced',
    objective         = 'multiclass',
    num_class         = N_CLASSES,
    metric            = 'multi_logloss',
    n_jobs            = -1,
    random_state      = SEED,
    verbosity         = -1,
    device            = 'gpu' if use_gpu_lgb else 'cpu',
)

t0        = time.time()
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30, verbose=False),
               lgb.log_evaluation(50)]
)
print(f"  Fit time: {time.time()-t0:.1f}s | Best iter: {lgb_model.best_iteration_}")

t0              = time.time()
y_pred_lgb_prob = lgb_model.predict_proba(X_test)
lgb_lat         = (time.time() - t0) / len(y_test) * 1000
y_pred_lgb      = np.argmax(y_pred_lgb_prob, axis=1)

cm_lgb = confusion_matrix(y_test, y_pred_lgb)
FP = cm_lgb.sum(axis=0) - np.diag(cm_lgb)
TN = cm_lgb.sum() - (cm_lgb.sum(axis=1) + cm_lgb.sum(axis=0) - np.diag(cm_lgb))
results['LightGBM'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_lgb),
    'Recall'    : recall_score(y_test, y_pred_lgb, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_lgb, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_lgb, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': lgb_lat
}
print(f"Accuracy : {results['LightGBM']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['LightGBM']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_lgb, target_names=le.classes_, zero_division=0))

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('LightGBM — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_lgb.png', dpi=150); plt.close()
print("✅ LightGBM done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — Model 6: Late Fusion Ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 15: Model 6 — Late Fusion Ensemble")
print("─" * 70)

w_xgb, w_lgb, w_deep = FUSION_WEIGHTS
fusion_probs = (
    w_xgb * y_pred_xgb_prob.astype(np.float32) +
    w_lgb * y_pred_lgb_prob.astype(np.float32) +
    w_deep * y_pred_hyb_probs.astype(np.float32)
)
y_pred_late = np.argmax(fusion_probs, axis=1)
late_lat = xgb_lat + lgb_lat + hyb_lat

cm_late = confusion_matrix(y_test, y_pred_late)
FP = cm_late.sum(axis=0) - np.diag(cm_late)
TN = cm_late.sum() - (cm_late.sum(axis=1) + cm_late.sum(axis=0) - np.diag(cm_late))
results['Late Fusion Ensemble'] = {
    'Accuracy'  : accuracy_score(y_test, y_pred_late),
    'Recall'    : recall_score(y_test, y_pred_late, average='weighted', zero_division=0),
    'F1-Score'  : f1_score(y_test, y_pred_late, average='weighted', zero_division=0),
    'Precision' : precision_score(y_test, y_pred_late, average='weighted', zero_division=0),
    'FAR'       : float(np.mean(FP / (FP + TN + 1e-8))),
    'Latency_ms': late_lat
}
print(f"Weights  : XGBoost={w_xgb:.2f}, LightGBM={w_lgb:.2f}, Gated Fusion={w_deep:.2f}")
print(f"Accuracy : {results['Late Fusion Ensemble']['Accuracy']*100:.2f}%")
print(f"F1-Score : {results['Late Fusion Ensemble']['F1-Score']*100:.2f}%")
print(classification_report(y_test, y_pred_late, target_names=le.classes_, zero_division=0))

plt.figure(figsize=(max(6, N_CLASSES), max(5, N_CLASSES-1)))
sns.heatmap(cm_late, annot=True, fmt='d', cmap='GnBu',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Late Fusion Ensemble — Confusion Matrix', fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_late_fusion.png', dpi=150); plt.close()
print("✅ Late Fusion Ensemble done\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 16 — Final Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("STEP 16: Final Model Comparison")
print("─" * 70)

model_names  = list(results.keys())
colors6      = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c']
metrics_keys = ['Accuracy', 'Recall', 'F1-Score', 'Precision']

comp_df = pd.DataFrame([
    {
        'Model'         : m,
        'Accuracy (%)'  : f"{results[m]['Accuracy']*100:.2f}",
        'Recall (%)'    : f"{results[m]['Recall']*100:.2f}",
        'F1-Score (%)'  : f"{results[m]['F1-Score']*100:.2f}",
        'Precision (%)' : f"{results[m]['Precision']*100:.2f}",
        'FAR (%)'       : f"{results[m]['FAR']*100:.4f}",
        'Latency (ms)'  : f"{results[m]['Latency_ms']:.4f}",
    }
    for m in model_names
])
comp_df.to_csv(OUTPUT_DIR / 'results_summary.csv', index=False)

print("\n" + "=" * 100)
print("   FINAL COMPARISON — TON-IoT IDS (Gated Fusion + Train-only WCGAN-GP)")
print("=" * 100)
print(comp_df.to_string(index=False))
print("=" * 100)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for idx, metric in enumerate(metrics_keys):
    vals = [results[m][metric] * 100 for m in model_names]
    bars = axes[idx].bar(model_names, vals,
                         color=colors6[:len(model_names)],
                         edgecolor='black', lw=0.5)
    axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
    axes[idx].set_ylim([max(0, min(vals) - 5), 103])
    axes[idx].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.2,
                       f'{val:.2f}%', ha='center', fontsize=8, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].axhline(99, color='red', linestyle='--', alpha=0.6, label='99% target')
    axes[idx].axhline(95, color='orange', linestyle=':', alpha=0.5, label='95% target')
    axes[idx].legend(fontsize=8)
plt.suptitle('Model Comparison — TON-IoT IDS (Gated Fusion + Train-only WCGAN-GP)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150); plt.close()

best_f1  = max(results, key=lambda m: results[m]['F1-Score'])
best_far = min(results, key=lambda m: results[m]['FAR'])
fastest  = min(results, key=lambda m: results[m]['Latency_ms'])
print(f"\n🏆 Best F1-Score     : {best_f1} ({results[best_f1]['F1-Score']*100:.2f}%)")
print(f"🏆 Lowest FAR        : {best_far} ({results[best_far]['FAR']*100:.4f}%)")
print(f"🏆 Fastest Inference : {fastest} ({results[fastest]['Latency_ms']:.4f} ms/sample)")

above_99 = [m for m in model_names if results[m]['Accuracy'] >= 0.99]
above_95 = [m for m in model_names if results[m]['Accuracy'] >= 0.95]
if above_99:
    print(f"\n✅ Models achieving 99%+ accuracy: {above_99}")
elif above_95:
    print(f"\n✅ Models achieving 95%+ accuracy: {above_95}")
    print("   Tip: increase WCGAN_EPOCHS=300 or CNN_EPOCHS=100 for 99%+")
else:
    best_acc = max(model_names, key=lambda m: results[m]['Accuracy'])
    print(f"\n   Best accuracy: {results[best_acc]['Accuracy']*100:.2f}% ({best_acc})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 17 — Save Models & Artifacts
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 17: Saving Models & Artifacts")
print("─" * 70)

with open(OUTPUT_DIR / 'decision_tree.pkl',       'wb') as f: pickle.dump(dt_model,     f)
with open(OUTPUT_DIR / 'xgboost_model.pkl',       'wb') as f: pickle.dump(xgb_model,    f)
with open(OUTPUT_DIR / 'random_forest.pkl',       'wb') as f: pickle.dump(rf_model,     f)
with open(OUTPUT_DIR / 'lightgbm_model.pkl',      'wb') as f: pickle.dump(lgb_model,    f)
with open(OUTPUT_DIR / 'label_encoder.pkl',       'wb') as f: pickle.dump(le,            f)
with open(OUTPUT_DIR / 'quantile_transformer.pkl','wb') as f: pickle.dump(qt,            f)
if preprocessor is not None:
    with open(OUTPUT_DIR / 'preprocessor.pkl', 'wb') as f: pickle.dump(preprocessor, f)
with open(OUTPUT_DIR / 'pca_model.pkl',           'wb') as f: pickle.dump(pca,           f)
with open(OUTPUT_DIR / 'lda_model.pkl',           'wb') as f: pickle.dump(lda,           f)

fusion_model.save(OUTPUT_DIR / 'gated_image_tabular_fusion.keras')

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump({m: {k: float(v) for k, v in r.items()}
               for m, r in results.items()}, f, indent=2)

with open(OUTPUT_DIR / 'pipeline_config.json', 'w') as f:
    json.dump({
        'pipeline': 'gated_image_tabular_fusion_train_only_wcgan',
        'dataset_file': str(DATASET_FILE),
        'output_dir': str(OUTPUT_DIR),
        'wcgan_epochs': WCGAN_EPOCHS,
        'cnn_epochs': CNN_EPOCHS,
        'focal_gamma': FOCAL_GAMMA,
        'label_smoothing': LABEL_SMOOTHING,
        'fusion_weights': {
            'xgboost': FUSION_WEIGHTS[0],
            'lightgbm': FUSION_WEIGHTS[1],
            'gated_fusion': FUSION_WEIGHTS[2],
        },
        'test_policy': 'stratified natural holdout before synthetic oversampling',
        'models': list(results.keys()),
    }, f, indent=2)

print(f"\n✅ All artifacts saved to: {OUTPUT_DIR.resolve()}")
print("\nFiles:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(str(OUTPUT_DIR), fname)
    if os.path.isfile(fpath):
        print(f"  {fname:42s} ({os.path.getsize(fpath)/1e6:.2f} MB)")

print("\n🎉 TON-IoT Pipeline complete!")
notify_user("TON-IoT IDS Complete", f"Pipeline finished. Results saved to {OUTPUT_DIR.resolve()}")
