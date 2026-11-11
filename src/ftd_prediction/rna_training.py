"""Optional, donor-isolated scVI training and expression baselines.

Importing this module does not import the neural stack. Private artifacts belong
on approved storage; this module has no disease target or checkpoint downloader.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import logging
import math
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np

from .contracts import is_resolved


_LOG = logging.getLogger(__name__)
_TOTAL = 10_000.0
_DEFAULTS = dict(
    seed=11, max_epochs=100, max_genes=2000, n_hidden=128, n_latent=16,
    n_layers=2, batch_size=256, learning_rate=0.001, weight_decay=0.000001,
    kl_warmup_epochs=20, accelerator="cpu", max_minutes=60.0, num_threads=1,
)


def _identifier(value):
    return (is_resolved(value) and value == value.strip()
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _donors(value):
    if not isinstance(value, (list, tuple)) or not all(_identifier(v) for v in value):
        raise ValueError("Donor splits require explicit sequences of resolved identifiers.")
    if len(set(value)) != len(value):
        raise ValueError("Donor splits contain repeated identifiers.")
    return list(value)


def _config(config):
    if not isinstance(config, Mapping):
        raise ValueError("RNA configuration must be a mapping.")
    if set(config) - (set(_DEFAULTS) | {"train_donors", "test_donors"}):
        raise ValueError("RNA configuration contains unsupported settings.")
    if not {"train_donors", "test_donors"} <= set(config):
        raise ValueError("Both donor splits must be explicitly supplied.")
    cfg = {**_DEFAULTS, **config}
    for key in ("train_donors", "test_donors"):
        cfg[key] = _donors(cfg[key])
    if not cfg["train_donors"] or set(cfg["train_donors"]) & set(cfg["test_donors"]):
        raise ValueError("Training donors must be nonempty and disjoint from test donors.")
    bounds = dict(seed=(0, 2**32 - 1), max_epochs=(1, 100), max_genes=(2, 2000),
                  n_hidden=(1, 4096), n_latent=(1, 512), n_layers=(1, 16),
                  batch_size=(2, 65536), kl_warmup_epochs=(1, 100), num_threads=(1, 128))
    for key, (lower, upper) in bounds.items():
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
            raise ValueError("RNA integer settings are outside supported bounds.")
        cfg[key] = int(value)
    for key, upper in (("learning_rate", 1.0), ("weight_decay", 1.0), ("max_minutes", 60.0)):
        value = cfg[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 < value <= upper):
            raise ValueError("RNA numeric settings must be finite, positive and bounded.")
        cfg[key] = float(value)
    if cfg["accelerator"] not in ("cpu", "gpu"):
        raise ValueError("Select an explicit CPU or GPU accelerator.")
    return cfg


def _raw_counts(adata):
    from scipy import sparse

    matrix = adata.layers["counts"]
    if not (sparse.issparse(matrix) or isinstance(matrix, np.ndarray)):
        raise ValueError("Counts must be an in-memory numeric matrix.")
    values = matrix.data if sparse.issparse(matrix) else matrix
    if (values.dtype.kind not in "iuf" or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values != np.floor(values))
            or np.any(values > 2**24)):
        raise ValueError("Counts must be finite nonnegative UMI integers representable in float32.")
    counts = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if np.any(counts.data > 2**24):
        raise ValueError("Combined counts exceed the supported exact float32 range.")
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    if not np.isfinite(libraries).all() or np.any(libraries <= 0):
        raise ValueError("Every input cell must have a finite nonzero count library.")
    return counts


def _log_normalize(counts):
    """Fixed per-cell exposure scaling; never estimates query-wide parameters."""
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    normalized = counts.multiply((_TOTAL / libraries)[:, None]).tocsr()
    np.log1p(normalized.data, out=normalized.data)
    return normalized


def _select_features(train_log, genes, max_genes):
    mean = np.asarray(train_log.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(train_log.multiply(train_log).mean(axis=0)).ravel() - mean**2, 0)
    order = np.lexsort((np.asarray(genes, dtype=str), -variance))
    selected = order[:min(max_genes, len(genes))]
    return selected, variance


def _prepare(adata, config):
    import anndata as ad
    import pandas as pd

    cfg = _config(config)
    if not isinstance(adata, ad.AnnData) or adata.isbacked:
        raise ValueError("RNA input must be an in-memory AnnData.")
    if (adata.n_obs < 3 or adata.n_vars < 2 or "counts" not in adata.layers
            or "donor_id" not in adata.obs or not adata.obs.columns.is_unique
            or not adata.obs_names.is_unique or not adata.var_names.is_unique
            or not all(_identifier(v) for v in adata.obs_names)
            or not all(_identifier(v) for v in adata.var_names)):
        raise ValueError("RNA input requires counts, unique names and explicit donor metadata.")
    donors = adata.obs["donor_id"].tolist()
    if not all(_identifier(v) for v in donors):
        raise ValueError("Every input cell requires a resolved donor mapping.")
    if set(donors) != set(cfg["train_donors"]) | set(cfg["test_donors"]):
        raise ValueError("Donor splits must exactly cover supplied donors, with no unused entries.")
    train_rows = np.flatnonzero(np.isin(donors, cfg["train_donors"]))
    test_rows = np.flatnonzero(np.isin(donors, cfg["test_donors"]))
    if len(train_rows) < 3:
        raise ValueError("RNA fitting requires at least three training cells.")
    batch_size = min(cfg["batch_size"], len(train_rows))
    if len(train_rows) % batch_size == 1:
        raise ValueError("The requested batch size leaves a singleton training batch.")
    counts = _raw_counts(adata)
    # Physically isolate training cells before any feature fitting.
    train_log = _log_normalize(counts[train_rows])
    genes = adata.var_names.tolist()
    selected, variance = _select_features(train_log, genes, cfg["max_genes"])
    selected_counts = counts[:, selected].astype(np.float32)
    if np.any(np.asarray(selected_counts.sum(axis=1)).ravel() <= 0):
        raise ValueError("Feature selection leaves an empty count library; revise the feature limit.")
    labels = None
    if "cell_type" in adata.obs:
        labels = []
        for label in adata.obs["cell_type"].tolist():
            if label is None or (not isinstance(label, str) and pd.api.types.is_scalar(label) and pd.isna(label)):
                labels.append(None)
            elif isinstance(label, str):
                label = label.strip()
                labels.append(label if _identifier(label) else None)
            else:
                raise ValueError("Cell-type annotations must be strings or missing.")
    # AnnData supplied to scVI deliberately has no donor, annotation or diagnosis
    # columns, nor any of the caller's additional matrices or private cell names.
    train = ad.AnnData(
        X=selected_counts[train_rows].copy(),
        obs=pd.DataFrame(index=[f"row_{i}" for i in train_rows]),
        var=pd.DataFrame(index=[genes[i] for i in selected]),
    )
    train.layers["counts"] = train.X.copy()
    return dict(config=cfg, counts=counts, selected_counts=selected_counts,
                train=train, train_log=train_log[:, selected].tocsr(),
                selected=selected, variance=variance, genes=genes, labels=labels,
                donors=np.asarray(donors), train_rows=train_rows, test_rows=test_rows,
                batch_size=batch_size)


def _fit_pca(train_log, n_latent, seed):
    from sklearn.decomposition import PCA

    n_components = min(n_latent, train_log.shape[0] - 1, train_log.shape[1] - 1)
    # ARPACK uses implicit sparse centering; it never creates a dense expression
    # matrix. Only the small components and cell-by-latent scores are dense.
    pca = PCA(n_components=n_components, svd_solver="arpack", random_state=seed)
    pca.fit(train_log)
    if not np.isfinite(pca.components_).all() or not np.isfinite(pca.explained_variance_).all():
        raise RuntimeError("PCA fitting produced nonfinite parameters.")
    return pca


def _pca_mse(matrix, scores, mean):
    energy = (float(matrix.multiply(matrix).sum())
              - 2 * float(np.asarray(matrix.sum(axis=0)).ravel() @ mean)
              + matrix.shape[0] * float(mean @ mean))
    return max(0.0, energy - float(np.square(scores).sum())) / math.prod(matrix.shape)


def _fit_centroids(embedding, labels, donors):
    """Equal weight per fitting donor within each annotated cell type."""
    categories = sorted({label for label in labels if label is not None})
    centers = []
    labels = np.asarray(labels, dtype=object)
    for category in categories:
        means = []
        for donor in sorted(set(donors)):
            mask = (labels == category) & (donors == donor)
            if mask.any():
                means.append(np.asarray(embedding[mask].mean(axis=0)).ravel())
        centers.append(np.mean(means, axis=0))
    return categories, np.asarray(centers, dtype=np.float64).reshape(len(categories), embedding.shape[1])


def _predict_centroids(embedding, categories, centers):
    """Compute distances in bounded batches, including sparse expression input."""
    from scipy import sparse

    predictions = []
    for start in range(0, embedding.shape[0], 1024):
        block = embedding[start:start + 1024]
        norms = (np.asarray(block.multiply(block).sum(axis=1)).ravel()
                 if sparse.issparse(block) else np.square(block).sum(axis=1))
        distances = norms[:, None] - 2 * np.asarray(block @ centers.T) + np.square(centers).sum(axis=1)
        if not np.isfinite(distances).all():
            raise RuntimeError("Annotation probe produced nonfinite distances.")
        predictions.extend(categories[i] for i in distances.argmin(axis=1))
    return predictions


def _agreement(truth, prediction, categories):
    """Unweighted class metrics include unsupported true classes with zero recall."""
    from sklearn.metrics import f1_score

    keep = [i for i, label in enumerate(truth) if label is not None]
    if not keep:
        return dict(n_labelled=0, macro_f1=None, balanced_accuracy=None, unsupported_classes=[])
    actual = [truth[i] for i in keep]
    predicted = [prediction[i] for i in keep]
    # balanced_accuracy_score warns for predicted-only classes; compute the
    # equivalent mean recall over observed truth classes directly instead.
    classes = sorted(set(actual))
    recalls = [sum(t == c and p == c for t, p in zip(actual, predicted)) / actual.count(c) for c in classes]
    return dict(n_labelled=len(keep),
                macro_f1=float(f1_score(actual, predicted, average="macro", zero_division=0)),
                balanced_accuracy=float(np.mean(recalls)),
                unsupported_classes=sorted(set(actual) - set(categories)))


def _diagnostics(embedding):
    if not np.isfinite(embedding).all():
        raise RuntimeError("RNA inference produced nonfinite embeddings.")
    centered = embedding - embedding.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    probabilities = energy[energy > 0] / energy.sum() if energy.sum() else np.array([])
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if len(probabilities) else 0.0
    return dict(mean_latent_variance=float(np.var(embedding, axis=0).mean()),
                rank=int(np.linalg.matrix_rank(centered)), effective_rank=effective_rank)


def _parameter_hash(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if name.startswith("z_encoder."):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_split(train_idx, val_idx, test_idx, n_rows):
    if (train_idx is None or val_idx is None or test_idx is None
            or not np.array_equal(np.sort(train_idx), np.arange(n_rows))
            or len(val_idx) or len(test_idx)):
        raise RuntimeError("The actual scVI splitter differs from the required all-training split.")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _fit_scvi(prepared, output):
    import anndata as ad
    import pandas as pd
    import scvi
    import torch
    from lightning.pytorch.callbacks import Callback, Timer
    from scvi.train import TrainingPlan

    cfg = prepared["config"]
    train = prepared["train"]
    losses = []

    class Evidence(Callback):
        def on_train_start(self, trainer, pl_module):
            splitter = trainer.datamodule
            _check_split(splitter.train_idx, splitter.val_idx, splitter.test_idx, train.n_obs)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("scVI optimization produced a nonfinite loss.")
            losses.append(dict(epoch=int(trainer.current_epoch), step=int(trainer.global_step), loss=value))

    previous_threads = torch.get_num_threads()
    previous_seed = scvi.settings.seed
    try:
        torch.set_num_threads(cfg["num_threads"])
        scvi.settings.seed = cfg["seed"]
        scvi.model.SCVI.setup_anndata(train, layer="counts", batch_key=None, labels_key=None)
        model = scvi.model.SCVI(
            train, n_hidden=cfg["n_hidden"], n_latent=cfg["n_latent"], n_layers=cfg["n_layers"],
            dropout_rate=0.1, dispersion="gene", gene_likelihood="nb", use_observed_lib_size=True,
        )
        before = _parameter_hash(model.module)
        timer = Timer(duration=timedelta(minutes=cfg["max_minutes"]), interval="step")
        plan_kwargs = dict(lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                           n_epochs_kl_warmup=cfg["kl_warmup_epochs"], reduce_lr_on_plateau=False)
        model.train(
            max_epochs=cfg["max_epochs"], accelerator=cfg["accelerator"], devices=1,
            batch_size=prepared["batch_size"], train_size=1.0, validation_size=0.0,
            shuffle_set_split=False, early_stopping=False,
            datasplitter_kwargs=dict(external_indexing=[np.arange(train.n_obs),
                                                       np.array([], dtype=int), np.array([], dtype=int)],
                                     num_workers=0, drop_last=False),
            plan_kwargs=plan_kwargs, callbacks=[Evidence(), timer],
            default_root_dir=str(output), enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            benchmark=False, deterministic=True, precision="32-true", log_every_n_steps=1,
        )
        _check_split(model.train_indices, model.validation_indices, model.test_indices, train.n_obs)
        steps = int(model.trainer.global_step)
        after = _parameter_hash(model.module)
        if not losses or steps <= 0 or before == after:
            raise RuntimeError("scVI did not establish finite-loss encoder weight updates.")
        if any(not torch.isfinite(p).all() for p in model.module.parameters()):
            raise RuntimeError("scVI parameters are nonfinite.")
        checkpoint_path = output / "optimizer.ckpt"
        model.trainer.save_checkpoint(checkpoint_path, weights_only=False)
        model.save(str(output / "scvi_model"), save_anndata=False)
        # Only deserialize the full checkpoint just produced in this new run.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("global_step") != steps or "epoch" not in checkpoint
                or not checkpoint.get("optimizer_states") or not checkpoint.get("loops")):
            raise RuntimeError("Full optimizer checkpoint is incomplete.")
        reloaded = scvi.model.SCVI.load(str(output / "scvi_model"), adata=train.copy(),
                                        accelerator=cfg["accelerator"], device="auto")
        if any(not torch.equal(p.detach().cpu(), dict(reloaded.module.named_parameters())[name].detach().cpu())
               for name, p in model.module.named_parameters()):
            raise RuntimeError("Inference export changed model parameters on reload.")
        restored_plan = TrainingPlan(reloaded.module, **plan_kwargs)
        restored_plan.load_state_dict(checkpoint["state_dict"], strict=True)
        restored_optimizer = restored_plan.configure_optimizers()["optimizer"]
        restored_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        state = restored_optimizer.state_dict()["state"]
        if not state or any(not torch.isfinite(value).all() for item in state.values()
                            for value in item.values() if torch.is_tensor(value)):
            raise RuntimeError("Optimizer state reload failed its finite-state check.")
        model.module.eval()
        reloaded.module.eval()
        train_embedding = model.get_latent_representation(batch_size=prepared["batch_size"])
        train_reloaded = reloaded.get_latent_representation(batch_size=prepared["batch_size"])
        if not np.allclose(train_embedding, train_reloaded, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Reloaded scVI inference disagrees with the fitted model.")
        embedding = np.empty((len(prepared["donors"]), cfg["n_latent"]), dtype=np.float32)
        embedding[prepared["train_rows"]] = train_reloaded
        test_rows = prepared["test_rows"]
        if len(test_rows):
            query = ad.AnnData(X=prepared["selected_counts"][test_rows].copy(),
                               obs=pd.DataFrame(index=[f"row_{i}" for i in test_rows]), var=train.var.copy())
            query.layers["counts"] = query.X.copy()
            # Ordinary fixed-weight inference: no query training, feature refit,
            # donor-as-batch correction, or scArches adaptation.
            heldout = reloaded.get_latent_representation(query, batch_size=prepared["batch_size"])
            original = model.get_latent_representation(query.copy(), batch_size=prepared["batch_size"])
            if not np.allclose(heldout, original, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Reloaded holdout inference disagrees with the fitted model.")
            embedding[test_rows] = heldout
        if _parameter_hash(reloaded.module) != after:
            raise RuntimeError("Fixed-weight inference changed the reloaded encoder.")
        evidence = dict(encoder_sha256_before=before, encoder_sha256_after=after,
                        training_steps=steps, finite_loss_steps=len(losses),
                        full_checkpoint_reload_verified=True,
                        actual_train_indices=np.asarray(model.train_indices).tolist(),
                        actual_validation_indices=np.asarray(model.validation_indices).tolist(),
                        actual_test_indices=np.asarray(model.test_indices).tolist(),
                        inference_reload_max_abs_error=float(np.max(np.abs(train_embedding - train_reloaded))),
                        observed_epochs=len({row["epoch"] for row in losses}),
                        fit_seconds=float(timer.time_elapsed("train")),
                        termination_reason=("max_minutes" if timer.time_elapsed("train") >= 60 * cfg["max_minutes"]
                                            else "max_epochs"),
                        exact_resume_verified=False)
        return embedding, losses, evidence
    finally:
        torch.set_num_threads(previous_threads)
        # Do not reseed global generators during cleanup. Restore the scvi
        # setting itself; the lead owns process-level RNG isolation/provenance.
        scvi.settings._seed = previous_seed


def run_rna(adata, config, output_dir) -> dict:
    """Fit training donors, infer heldout donors, and save private evidence.

    ``output_dir`` must be new. Failed runs retain their partial artifacts and
    never emit a completed summary. The minute limit bounds training between
    batches, not preprocessing, inference or filesystem operations.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("RNA output directory must not already exist.")
    prepared = _prepare(adata, config)
    cfg = prepared["config"]
    # Resolve optional packages before creating a run directory.
    try:
        package_versions = {name: version(name) for name in
                            ("scvi-tools", "torch", "lightning", "anndata", "scipy", "scikit-learn", "numpy")}
    except ImportError:
        raise ImportError("RNA training requires the isolated optional neural environment.") from None
    if package_versions["scvi-tools"] != "1.5.2":
        raise RuntimeError("This backend requires the pinned scvi-tools 1.5.2 API.")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    _LOG.info("Starting RNA representation fit on explicit training donors.")
    train_rows, test_rows = prepared["train_rows"], prepared["test_rows"]
    selected = prepared["selected"]
    pca = _fit_pca(prepared["train_log"], cfg["n_latent"], cfg["seed"])
    # Query cells are normalized independently using the fixed exposure recipe;
    # they cannot influence ranking or PCA parameters.
    expression = _log_normalize(prepared["counts"])[:, selected].tocsr()
    pca_embedding = pca.transform(expression)
    scvi_embedding, history, evidence = _fit_scvi(prepared, output)
    metrics = dict(optimization=evidence,
                   pca_train_log_expression_mse=_pca_mse(expression[train_rows], pca_embedding[train_rows], pca.mean_),
                   pca_test_log_expression_mse=(_pca_mse(expression[test_rows], pca_embedding[test_rows], pca.mean_)
                                                if len(test_rows) else None),
                   scvi_embedding=_diagnostics(scvi_embedding[train_rows]),
                   pca_embedding=_diagnostics(pca_embedding[train_rows]))
    artifacts = dict(inference_model="scvi_model", optimizer_checkpoint="optimizer.ckpt",
                     embeddings="embeddings.npz", pca_model="pca_model.npz", selected_genes="selected_genes.json",
                     split="split.json", history="history.json", evidence="evidence.json", summary="summary.json")
    probe_arrays = {}
    if prepared["labels"] is not None:
        labels = prepared["labels"]
        train_labels = [labels[i] for i in train_rows]
        metrics["annotation_agreement"] = dict(scope="annotation_agreement_only", per_representation={})
        for name, matrix in (("expression", expression), ("pca", pca_embedding), ("scvi", scvi_embedding)):
            categories, centers = _fit_centroids(matrix[train_rows], train_labels, prepared["donors"][train_rows])
            probe_arrays[name + "_centroids"] = centers
            probe_arrays[name + "_categories"] = np.asarray(categories, dtype=str)
            if not categories:
                metrics["annotation_agreement"]["per_representation"][name] = dict(status="no_training_annotations")
                continue
            prediction = _predict_centroids(matrix, categories, centers)
            per_donor = {}
            for donor in cfg["test_donors"]:
                rows = np.flatnonzero(prepared["donors"] == donor)
                per_donor[donor] = _agreement([labels[i] for i in rows], [prediction[i] for i in rows], categories)
            metrics["annotation_agreement"]["per_representation"][name] = dict(
                training_resubstitution=_agreement(train_labels, [prediction[i] for i in train_rows], categories),
                heldout_per_donor=per_donor,
            )
        np.savez_compressed(output / "annotation_probes.npz", **probe_arrays)
        artifacts["annotation_probes"] = "annotation_probes.npz"
    np.savez_compressed(output / "embeddings.npz", scvi=scvi_embedding, pca=pca_embedding,
                        row_indices=np.arange(adata.n_obs))
    np.savez_compressed(output / "pca_model.npz", components=pca.components_, mean=pca.mean_,
                        explained_variance=pca.explained_variance_, selected_indices=selected)
    genes = [prepared["genes"][i] for i in selected]
    _write_json(output / "selected_genes.json", dict(
        indices=selected.tolist(), names=genes, fitting_donors=cfg["train_donors"],
        recipe="train_cell_log1p_library_10000_population_variance_descending_gene_name_ties",
        variances=prepared["variance"][selected].tolist(), normalization_total=_TOTAL,
        scvi_input="raw_selected_counts_observed_selected_library_size",
        baseline_input="log1p_selected_genes_normalized_by_all_input_genes",
    ))
    _write_json(output / "split.json", dict(train_donors=cfg["train_donors"], test_donors=cfg["test_donors"],
                                           train_rows=train_rows.tolist(), test_rows=test_rows.tolist(),
                                           cell_names=adata.obs_names.tolist(), validation_rows=[],
                                           actual_scvi_training_indices=evidence["actual_train_indices"],
                                           actual_scvi_validation_indices=evidence["actual_validation_indices"],
                                           actual_scvi_test_indices=evidence["actual_test_indices"]))
    _write_json(output / "history.json", history)
    _write_json(output / "evidence.json", evidence)
    result = dict(method="scvi", status="completed", train_donors=cfg["train_donors"],
                  test_donors=cfg["test_donors"], selected_genes=genes,
                  encoder_updated=True, reload_verified=True, training_steps=evidence["training_steps"],
                  metrics=metrics, artifacts=artifacts, config=cfg, effective_batch_size=prepared["batch_size"],
                  versions=package_versions, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_seconds=float(time.monotonic() - started),
                  evidence_scope="donor_heldout_technical_representation" if len(test_rows) else "descriptive_all_donor_fit")
    _write_json(output / "summary.json", result)
    _LOG.info("RNA representation artifacts saved; no disease model was fitted.")
    return result
