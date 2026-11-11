"""Optional, offline Novae 1.1.3 training and donor-held-out technical checks.

Importing this module does not import the optional neural stack. Call through the
lead runner on approved compute/storage; outputs contain private cell artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import logging
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time

import numpy as np


_DEFAULTS = dict(mode="frozen", seed=11, max_epochs=10, batch_size=256,
                 num_workers=0, max_minutes=45.0, accelerator="cpu",
                 radius_um=18.0, domain_level=7, max_genes=2000,
                 learning_rate=1e-4, num_threads=1)
_MISSING = {"", "unknown", "unresolved", "conflicting", "none", "null", "nan",
            "not_applicable", "restricted"}


def _integer(value, key, minimum=1, maximum=100000):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{key} must be a bounded integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be a bounded integer")
    return int(value)


def _positive(value, key, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{key} must be positive and finite")
    return float(value)


def _identifiers(values, key, unique=False):
    result = list(values)
    if any(not isinstance(x, str) or x.strip().lower() in _MISSING for x in result):
        raise ValueError(f"{key} must contain explicit string identifiers")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{key} must be unique")
    return result


def _validate(adata, config):
    """Validate all mappings/counts before constructing models or artifacts."""
    from scipy import sparse
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    allowed = set(_DEFAULTS) | {"train_donors", "test_donors", "checkpoint_path"}
    if set(config) - allowed:
        raise ValueError("Unsupported spatial configuration keys")
    cfg = {**_DEFAULTS, **config}
    if cfg["mode"] not in ("synthetic_smoke", "frozen", "fine_tune"):
        raise ValueError("Unsupported spatial mode")
    for key, lo, hi in (("seed", 0, 2**32 - 1), ("max_epochs", 1, 1000),
                        ("batch_size", 2, 4096), ("num_workers", 0, 32),
                        ("domain_level", 1, 65535), ("max_genes", 20, 50000),
                        ("num_threads", 1, 64)):
        cfg[key] = _integer(cfg[key], key, lo, hi)
    for key, hi in (("max_minutes", 1440), ("radius_um", 1000000),
                    ("learning_rate", 1)):
        cfg[key] = _positive(cfg[key], key, hi)
    if cfg["accelerator"] not in ("cpu", "gpu", "cuda"):
        raise ValueError("accelerator must explicitly select cpu or gpu")
    if cfg["mode"] == "synthetic_smoke":
        if cfg["accelerator"] != "cpu" or cfg["max_epochs"] > 2:
            raise ValueError("Synthetic smoke requires CPU and at most two epochs per phase")
        if cfg["domain_level"] > 8:
            raise ValueError("Synthetic domain_level exceeds the eight-prototype hierarchy")
        if adata.n_vars < 32 or cfg["max_genes"] < 32:
            raise ValueError("Pinned synthetic smoke requires at least thirty-two selected genes")
        if cfg.get("checkpoint_path") is not None:
            raise ValueError("Synthetic smoke does not accept external checkpoints")
        if adata.n_obs > 512 or adata.n_vars > 128:
            raise ValueError("Synthetic smoke requires tiny invented inputs")
    else:
        checkpoint = cfg.get("checkpoint_path")
        if not isinstance(checkpoint, (str, Path)) or not Path(checkpoint).is_dir():
            raise ValueError("A local checkpoint directory is required")
        checkpoint = Path(checkpoint).resolve()
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
            raise ValueError("Checkpoint requires local config.json and model.safetensors")
        cfg["checkpoint_path"] = str(checkpoint)
    if adata.n_obs == 0 or adata.n_vars < 20:
        raise ValueError("Spatial input requires cells and at least twenty measured genes")
    _identifiers(adata.obs_names, "Cell names", unique=True)
    genes = _identifiers(adata.var_names, "Gene names", unique=True)
    if len(set(x.lower() for x in genes)) != len(genes):
        raise ValueError("Gene names must remain unique after Novae case conversion")
    for key in ("donor_id", "section_id"):
        if key not in adata.obs:
            raise ValueError("Required donor/section mapping is missing")
        _identifiers(adata.obs[key], "Donor/section mapping")
    if "tissue_component" in adata.obs:
        _identifiers(adata.obs["tissue_component"], "Component mapping")
    if (adata.obs.groupby("section_id", observed=True)["donor_id"].nunique() > 1).any():
        raise ValueError("A physical section must map to exactly one donor")
    split = []
    for key in ("train_donors", "test_donors"):
        if key not in cfg or not isinstance(cfg[key], (list, tuple)):
            raise ValueError("Explicit train_donors and test_donors lists are required")
        cfg[key] = _identifiers(cfg[key], "Donor split", unique=True)
        split.append(set(cfg[key]))
    if not split[0] or split[0] & split[1] or split[0] | split[1] != set(adata.obs["donor_id"]):
        raise ValueError("Donor splits must be disjoint and exactly cover supplied donors")
    xy = np.asarray(adata.obsm.get("spatial"), dtype=float)
    if xy.shape != (adata.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError("Spatial coordinates must be finite two-dimensional micron values")
    if "counts" not in adata.layers:
        raise ValueError("Measured unnormalized counts layer is required")
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or np.any(counts.data < 0)
            or np.any(counts.data != np.floor(counts.data))):
        raise ValueError("Counts must contain finite nonnegative measured integers")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Every input cell must have a finite positive count total")
    train_mask = adata.obs["donor_id"].isin(cfg["train_donors"]).to_numpy()
    if train_mask.sum() < max(8, cfg["batch_size"]):
        raise ValueError("Training subset is too small for the requested batch size")
    return cfg, counts, train_mask


def _normalize(counts):
    """Fixed 10,000-count scaling and log1p; no fitted or pooled statistics."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    x = counts.multiply((10000.0 / totals)[:, None]).tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    return x.astype(np.float32)


def _select_features(train_x, genes, max_genes, vocabulary=None):
    mean = np.asarray(train_x.mean(axis=0)).ravel()
    variance = np.maximum(0, np.asarray(train_x.power(2).mean(axis=0)).ravel() - mean**2)
    eligible = [i for i, gene in enumerate(genes)
                if vocabulary is None or gene.lower() in vocabulary]
    selected = sorted(eligible, key=lambda i: (-float(variance[i]), genes[i]))[:max_genes]
    if len(selected) < 20:
        raise ValueError("Fewer than twenty selected genes overlap the offline vocabulary")
    return np.asarray(selected, dtype=int)


def _make_sections(adata, counts, normalized, selected, mask):
    """Create new minimal objects: never carry caller graphs/diagnoses/caches."""
    import anndata as ad
    import pandas as pd
    components = adata.obs.get("tissue_component")
    groups = {}
    for i in np.flatnonzero(mask):
        key = (adata.obs["section_id"].iloc[i], None if components is None else components.iloc[i])
        groups.setdefault(key, []).append(i)
    sections = []
    for rows in groups.values():
        rows = np.asarray(rows)
        xy = np.asarray(adata.obsm["spatial"])[rows].copy()
        if len(rows) < 4 or len(np.unique(xy, axis=0)) != len(rows):
            raise ValueError("Each section/component needs at least four distinct cell coordinates")
        if np.linalg.matrix_rank(xy - xy.mean(axis=0)) != 2:
            raise ValueError("Section/component coordinates must support a two-dimensional graph")
        obs = adata.obs.iloc[rows][["donor_id", "section_id"]].copy()
        if components is not None:
            obs["tissue_component"] = components.iloc[rows].to_numpy()
        if "cell_type" in adata.obs:
            obs["cell_type"] = adata.obs["cell_type"].iloc[rows].to_numpy()
        obs["input_row"] = rows
        section = ad.AnnData(X=normalized[rows][:, selected].copy(), obs=obs,
                            var=pd.DataFrame(index=adata.var_names[selected].copy()))
        section.layers["counts"] = counts[rows][:, selected].copy()
        section.var["highly_variable"] = True
        section.obsm["spatial"] = xy
        sections.append(section)
    return sections


def _audit_graphs(sections, radius_um):
    metrics = dict(cross_section_edges=0, cross_donor_edges=0,
                   cross_component_edges=0, directed_edges=0, max_distance_error_um=0.0)
    for section in sections:
        graph = section.obsp["spatial_distances"].tocoo()
        if not graph.nnz or not np.isfinite(graph.data).all() or np.any(graph.data <= 0):
            raise ValueError("Every section/component needs finite positive distance edges")
        for column, metric in (("section_id", "cross_section_edges"),
                               ("donor_id", "cross_donor_edges"),
                               ("tissue_component", "cross_component_edges")):
            if column in section.obs:
                values = section.obs[column].to_numpy()
                metrics[metric] += int(np.count_nonzero(values[graph.row] != values[graph.col]))
        xy = np.asarray(section.obsm["spatial"])
        expected = np.linalg.norm(xy[graph.row] - xy[graph.col], axis=1)
        if not np.allclose(expected, graph.data, rtol=1e-6, atol=1e-6):
            raise ValueError("Graph edge weights do not match micron distances")
        if np.any(graph.row == graph.col) or np.any(expected > radius_um + 1e-6):
            raise ValueError("Graph contains self edges or edges beyond the declared radius")
        metrics["max_distance_error_um"] = max(metrics["max_distance_error_um"],
                                               float(np.max(np.abs(expected - graph.data))))
        metrics["directed_edges"] += int(graph.nnz)
    if any(metrics[k] for k in ("cross_section_edges", "cross_donor_edges", "cross_component_edges")):
        raise ValueError("Graph crosses a donor, section or tissue-component boundary")
    return metrics


def _state_hash(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Model state contains nonfinite parameters or buffers")
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Spatial runtime ceiling reached; partial artifacts are not a completed fit")


@contextmanager
def _private_runtime(output):
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    original_stderr = sys.stderr

    def stream_handlers():
        loggers = [root] + [x for x in logging.Logger.manager.loggerDict.values()
                            if isinstance(x, logging.Logger)]
        return {handler for logger in loggers for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)}

    with open(output / "runtime.log", "w") as log:
        handler = logging.StreamHandler(log)
        root.handlers = [handler]
        # Novae installs a non-propagating console logger. Redirect existing
        # handlers too, including when another caller imported Novae first.
        streams = {h: h.stream for h in stream_handlers() if h is not handler}
        for h in streams:
            h.setStream(log)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                yield
        finally:
            for h, stream in streams.items():
                h.setStream(stream)
            # Handlers first created by optional imports must not retain our
            # soon-to-close private file after returning to the caller.
            for h in stream_handlers() - set(streams) - {handler}:
                if h.stream is log:
                    h.setStream(original_stderr)
            root.handlers = old_handlers
            handler.close()


@contextmanager
def _novae_settings(novae, torch, num_threads):
    from novae._constants import Nums
    old = (novae.settings.auto_preprocessing, novae.settings.disable_multimodal,
           novae.settings.scale_to_microns, Nums.LAZY_LOADING_SIZE_THRESHOLD,
           torch.get_num_threads())
    novae.settings.auto_preprocessing = False
    novae.settings.disable_multimodal = True
    novae.settings.scale_to_microns = 1.0
    # Upstream's default caches a dense section matrix below 5e8 elements.
    # Force neighborhood-level conversion even for smaller real inputs.
    novae.settings.enable_lazy_loading(0)
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        (novae.settings.auto_preprocessing, novae.settings.disable_multimodal,
         novae.settings.scale_to_microns, Nums.LAZY_LOADING_SIZE_THRESHOLD, threads) = old
        torch.set_num_threads(threads)


def _fit(model, train, cfg, output, phase, deadline, fine_tune):
    import torch
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint

    class Evidence(Callback):
        def __init__(self):
            self.losses = []
            self.epochs = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            _check_time(deadline)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is None or not torch.isfinite(loss).all():
                raise ValueError("Novae produced a nonfinite training loss")
            self.losses.append(float(loss.detach().cpu()))

        def on_train_epoch_end(self, trainer, pl_module):
            self.epochs.append(int(trainer.current_epoch))

    evidence = Evidence()
    checkpoint = ModelCheckpoint(dirpath=output / phase, every_n_epochs=1,
                                 save_top_k=-1, save_last=True, save_weights_only=False)
    _check_time(deadline)
    kwargs = dict(max_epochs=cfg["max_epochs"], accelerator=cfg["accelerator"],
                  num_workers=cfg["num_workers"], lr=cfg["learning_rate"],
                  min_delta=0.0, patience=3, use_scheduler=False, devices=1,
                  precision="32-true", callbacks=[evidence, checkpoint], logger=False,
                  enable_progress_bar=False, enable_model_summary=False,
                  max_time={"seconds": max(1, int(deadline - time.monotonic()))},
                  default_root_dir=str(output))
    if cfg["mode"] == "synthetic_smoke":
        kwargs["limit_train_batches"] = 2
    if fine_tune:
        model.fine_tune(train, reference="all", min_prototypes_ratio=1.0, **kwargs)
    else:
        model.fit(**kwargs)
    _check_time(deadline)
    steps = int(model.trainer.global_step)
    if steps <= 0 or not evidence.losses or not checkpoint.last_model_path:
        raise ValueError("Novae did not complete finite-loss optimization and checkpointing")
    return {"phase": phase, "training_steps": steps, "losses": evidence.losses,
            "completed_epochs": len(evidence.epochs),
            "termination": "max_epochs" if len(evidence.epochs) == cfg["max_epochs"] else "training_loss_early_stop",
            "checkpoint": str(Path(checkpoint.last_model_path).relative_to(output))}


def _infer(model, sections, cfg, deadline):
    if model.mode.zero_shot:
        raise ValueError("Persistent zero-shot mode is forbidden for held-out mapping")
    model.eval()
    before = _state_hash(model)
    for section in sections:
        _check_time(deadline)
        model.compute_representations(section, zero_shot=False, accelerator=cfg["accelerator"],
                                      num_workers=cfg["num_workers"])
        model.assign_domains(section, level=cfg["domain_level"])
    if _state_hash(model) != before or model.mode.zero_shot:
        raise ValueError("Inference unexpectedly changed model state")


def _latent_metrics(sections, domain_level):
    if not sections:
        return {"cells": 0, "invalid_neighborhoods": 0, "latent_variance": None,
                "latent_rank": None, "domain_occupancy": {}, "annotation_agreement": None}
    valid_latents = []
    invalid = 0
    occupancy = {}
    annotation_labels, annotation_domains = [], []
    for section in sections:
        valid = section.obs["neighborhood_valid"].to_numpy(dtype=bool)
        latent = np.asarray(section.obsm["novae_latent"])
        if not np.isfinite(latent).all() or not valid.any():
            raise ValueError("A section has no valid finite neural neighborhoods")
        invalid += int((~valid).sum())
        valid_latents.append(latent[valid])
        for key, count in section.obs.loc[valid, f"novae_domains_{domain_level}"].value_counts().items():
            occupancy[str(key)] = occupancy.get(str(key), 0) + int(count)
        if "cell_type" in section.obs:
            for label, domain in zip(section.obs.loc[valid, "cell_type"],
                                     section.obs.loc[valid, f"novae_domains_{domain_level}"]):
                if isinstance(label, str) and label.strip().lower() not in _MISSING:
                    annotation_labels.append(label)
                    annotation_domains.append(str(domain))
    latent = np.concatenate(valid_latents)
    centered = latent - latent.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.count_nonzero(singular > (singular[0] * max(centered.shape) * np.finfo(latent.dtype).eps)))
    agreement = None
    if annotation_labels:
        from sklearn.metrics import adjusted_rand_score
        agreement = {"labelled_valid_cells": len(annotation_labels),
                     "adjusted_rand_index": float(adjusted_rand_score(annotation_labels, annotation_domains)),
                     "scope": "descriptive_cell_annotation_agreement_labels_not_verified_truth"}
    return {"cells": sum(s.n_obs for s in sections), "invalid_neighborhoods": invalid,
            "valid_neighborhoods": len(latent),
            "latent_variance": [float(x) for x in latent.var(axis=0)],
            "latent_rank": rank, "domain_occupancy": occupancy,
            "collapsed": bool(rank == 0 or len([v for v in occupancy.values() if v]) <= 1),
            "annotation_agreement": agreement}


def _expression_baseline(normalized, selected, train_mask, adata, output, seed):
    """Training-fitted expression-only SVD, without a spatial graph or labels."""
    from sklearn.decomposition import TruncatedSVD
    x = normalized[:, selected]
    n_components = min(8, len(selected) - 1, int(train_mask.sum()) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(x[train_mask])
    embeddings = svd.transform(x)
    if not np.isfinite(embeddings).all():
        raise ValueError("Expression baseline produced nonfinite values")
    np.savez_compressed(output / "expression_baseline.npz", embeddings=embeddings,
                        components=svd.components_, input_rows=np.arange(adata.n_obs))
    metrics = {"method": "training_fitted_log_normalized_uncentered_svd",
               "components": n_components,
               "training_explained_variance_ratio": [float(v) for v in svd.explained_variance_ratio_],
               "annotation_agreement": None}
    if "cell_type" in adata.obs:
        labels = adata.obs["cell_type"]
        labelled = np.array([isinstance(v, str) and v.strip().lower() not in _MISSING for v in labels])
        types = sorted(set(labels[train_mask & labelled]))
        if types:
            centroids = np.stack([embeddings[train_mask & labels.eq(label).fillna(False).to_numpy(dtype=bool)].mean(axis=0)
                                  for label in types])
            scores = {}
            for name, mask in (("train", train_mask), ("test", ~train_mask)):
                rows = mask & labelled
                # These are cell annotation agreements, not independent donor disease metrics.
                if rows.any():
                    dist = ((embeddings[rows, None] - centroids[None])**2).sum(axis=2)
                    prediction = np.asarray(types)[dist.argmin(axis=1)]
                    scores[name] = {"labelled_cells": int(rows.sum()),
                                    "accuracy": float(np.mean(prediction == labels.to_numpy()[rows]))}
                else:
                    scores[name] = None
            metrics["annotation_agreement"] = scores
            np.savez_compressed(output / "annotation_centroids.npz", centroids=centroids, labels=np.asarray(types))
    return metrics


def _execute(adata, cfg, counts, train_mask, output, started):
    import importlib.metadata
    import lightning as L
    import novae
    import torch
    if novae.__version__ != "1.1.3":
        raise ValueError("This backend requires the verified Novae 1.1.3 API")
    deadline = started + 60 * cfg["max_minutes"]
    L.seed_everything(cfg["seed"], workers=True, verbose=False)
    model = None
    checkpoint_hashes = None
    if cfg["mode"] != "synthetic_smoke":
        path = Path(cfg["checkpoint_path"])
        checkpoint_hashes = {name: _file_hash(path / name) for name in ("config.json", "model.safetensors")}
        model = novae.Novae.from_pretrained(path, local_files_only=True, map_location="cpu", strict=True)
        if model.cell_embedder is None or model.hparams.embedding_name is not None:
            raise ValueError("This backend requires an expression checkpoint")
        model.hparams.batch_size = cfg["batch_size"]
    vocabulary = None if model is None else set(model.cell_embedder.gene_names)
    normalized = _normalize(counts)
    selected = _select_features(normalized[train_mask], list(adata.var_names), cfg["max_genes"], vocabulary)
    train = _make_sections(adata, counts, normalized, selected, train_mask)
    test = _make_sections(adata, counts, normalized, selected, ~train_mask)
    novae.spatial_neighbors(train + test, slide_key="section_id", radius=cfg["radius_um"],
                            coord_type="generic", delaunay=True, verbose=False)
    graph_metrics = _audit_graphs(train + test, cfg["radius_um"])
    histories = []
    scratch_evidence = None
    if model is None:
        model = novae.Novae(train, var_names=list(adata.var_names[selected]), embedding_size=8,
                            hidden_size=16, output_size=8, num_layers=2, heads=2,
                            num_prototypes=8, batch_size=cfg["batch_size"],
                            n_hops_local=1, n_hops_view=1, min_prototypes_ratio=1.0)
        scratch_before = _state_hash(model.encoder)
        histories.append(_fit(model, train, cfg, output, "scratch_checkpoints", deadline, False))
        scratch_evidence = {"encoder_before": scratch_before, "encoder_after": _state_hash(model.encoder)}
        if scratch_evidence["encoder_before"] == scratch_evidence["encoder_after"]:
            raise ValueError("Synthetic scratch fitting did not update encoder weights")
        model.save_pretrained(output / "scratch_model")
        model = novae.Novae.from_pretrained(output / "scratch_model", local_files_only=True, map_location="cpu", strict=True)
        if _state_hash(model.encoder) != scratch_evidence["encoder_after"]:
            raise ValueError("Scratch encoder did not reload exactly")
    if cfg["domain_level"] > model.swav_head.num_prototypes:
        raise ValueError("domain_level exceeds the checkpoint prototype hierarchy")
    encoder_before = _state_hash(model.encoder)
    embedder_before = _state_hash(model.cell_embedder)
    prototypes_before = _state_hash(model.swav_head)
    model.eval()
    if cfg["mode"] == "frozen":
        model.to(torch.device("cpu" if cfg["accelerator"] == "cpu" else "cuda"))
        model.num_workers = cfg["num_workers"]
        model.init_prototypes(train, reference="all")
    else:
        histories.append(_fit(model, train, cfg, output, "fine_tune_checkpoints", deadline, True))
    encoder_after = _state_hash(model.encoder)
    updated = encoder_before != encoder_after
    if updated != (cfg["mode"] != "frozen"):
        raise ValueError("Encoder weight change does not match the declared mode")
    if cfg["mode"] == "frozen" and embedder_before != _state_hash(model.cell_embedder):
        raise ValueError("Frozen prototype fitting changed expression embedding weights")
    # Build hierarchy from train-fitted prototypes, then persist the exact mapping.
    hierarchy = model.swav_head.clusters_levels.copy()
    _infer(model, train, cfg, deadline)
    train_latents = [s.obsm["novae_latent"].copy() for s in train]
    train_domains = [s.obs[f"novae_domains_{cfg['domain_level']}"].astype(str).to_numpy() for s in train]
    model.save_pretrained(output / "model")
    saved_hash = _state_hash(model)
    prototypes_after = _state_hash(model.swav_head)
    reloaded = novae.Novae.from_pretrained(output / "model", local_files_only=True, map_location="cpu", strict=True)
    if _state_hash(reloaded) != saved_hash or not np.array_equal(hierarchy, reloaded.swav_head.clusters_levels):
        raise ValueError("Reloaded weights or prototype hierarchy differ")
    _infer(reloaded, train, cfg, deadline)
    reload_error = max(float(np.max(np.abs(s.obsm["novae_latent"] - previous)))
                       for s, previous in zip(train, train_latents))
    if reload_error > 1e-5 or any(not np.array_equal(previous, s.obs[f"novae_domains_{cfg['domain_level']}"].astype(str).to_numpy())
                                 for s, previous in zip(train, train_domains)):
        raise ValueError("Reloaded training representations or domains differ")
    _infer(reloaded, test, cfg, deadline)
    metrics = {**graph_metrics, "train": _latent_metrics(train, cfg["domain_level"]),
               "test": _latent_metrics(test, cfg["domain_level"]),
               "masks_checked": False, "masks_status": "unavailable_not_checked",
               "test_section_adaptation": bool(test), "reload_max_absolute_error": reload_error}
    metrics["expression_baseline"] = _expression_baseline(normalized, selected, train_mask, adata, output, cfg["seed"])
    artifact_sections = train + test
    # Only small latent matrices are dense; expression/count matrices remain sparse.
    np.savez_compressed(output / "embeddings.npz",
                        embeddings=np.concatenate([s.obsm["novae_latent"] for s in artifact_sections]),
                        input_rows=np.concatenate([s.obs["input_row"].to_numpy() for s in artifact_sections]),
                        valid=np.concatenate([s.obs["neighborhood_valid"].to_numpy() for s in artifact_sections]),
                        domains=np.concatenate([s.obs[f"novae_domains_{cfg['domain_level']}"].astype(str).to_numpy() for s in artifact_sections]))
    artifacts = dict(model="model", embeddings="embeddings.npz", expression_baseline="expression_baseline.npz",
                     runtime_log="runtime.log", provenance="provenance.json", training_history="training_history.json",
                     summary="summary.json")
    for history in histories:
        artifacts[history["phase"]] = history["checkpoint"]
    if cfg["mode"] == "synthetic_smoke":
        artifacts["scratch_model"] = "scratch_model"
    if (output / "annotation_centroids.npz").exists():
        artifacts["annotation_centroids"] = "annotation_centroids.npz"
    provenance = {"config": cfg, "selected_indices": selected.tolist(),
                  "selected_genes": list(adata.var_names[selected]),
                  "feature_fit_donors": cfg["train_donors"], "prototype_fit_donors": cfg["train_donors"],
                  "normalization": "per-cell total=10000 then log1p, using all measured genes before selection",
                  "feature_selection": "training-only log-normalized variance, ties by original gene name",
                  "coordinate_units": "microns", "coordinate_transform": "identity",
                  "graphs": "Delaunay pruned by radius, separate physical section and component; no degree cap",
                  "section_mapping": [{"section_id": s.obs["section_id"].iloc[0],
                                       "donor_id": s.obs["donor_id"].iloc[0],
                                       "component": s.obs["tissue_component"].iloc[0] if "tissue_component" in s.obs else None,
                                       "cells": s.n_obs} for s in artifact_sections],
                  "checkpoint_hashes": checkpoint_hashes, "encoder_before": encoder_before,
                  "encoder_after": encoder_after, "prototypes_before": prototypes_before,
                  "prototypes_after": prototypes_after, "scratch": scratch_evidence,
                  "domain_level": cfg["domain_level"],
                  "upstream_prototype_kmeans_seed": 0,
                  "leaf_domain_mapping": hierarchy[-cfg["domain_level"]].tolist(),
                  "holdout_inference_state_unchanged": True,
                  "exact_resume_supported": False, "source_sha256": _file_hash(Path(__file__)),
                  "counts_sha256": _matrix_hash(counts),
                  "versions": {n: importlib.metadata.version(n) for n in ("novae", "torch", "lightning", "anndata", "scipy", "scikit-learn")}}
    _write_json(output / "provenance.json", provenance)
    _write_json(output / "training_history.json", histories)
    _check_time(deadline)
    result = dict(method="novae", status="completed", mode=cfg["mode"],
                  train_donors=cfg["train_donors"], test_donors=cfg["test_donors"],
                  selected_genes=list(adata.var_names[selected]), encoder_updated=updated,
                  reload_verified=True, training_steps=sum(h["training_steps"] for h in histories),
                  metrics=metrics, artifacts=artifacts, elapsed_seconds=time.monotonic() - started,
                  scope="descriptive_all_donor_fit" if not test else "donor_held_out_technical_comparison_with_test_section_adaptation")
    _write_json(output / "summary.json", result)
    return result


def _matrix_hash(matrix):
    digest = hashlib.sha256(str(matrix.shape).encode())
    for value in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_spatial(adata, config, output_dir) -> dict:
    """Fit only explicit train donors and save locally; never download weights.

    The output directory must be new. Failures retain private diagnostic/partial
    artifacts and raise a generic message without donor, cell or path values.
    ``synthetic_smoke`` is restricted to tiny invented inputs, CPU, two epochs
    and two batches per phase. It cannot verify that caller data are invented.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Spatial output directory must not already exist")
    started = time.monotonic()
    cfg, counts, train_mask = _validate(adata, config)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    with _private_runtime(output):
        try:
            import novae
            import torch
            with _novae_settings(novae, torch, cfg["num_threads"]):
                return _execute(adata, cfg, counts, train_mask, output, started)
        except Exception:
            logging.exception("Spatial backend failed; private diagnostic follows")
            raise RuntimeError("Spatial backend failed; inspect private runtime.log") from None
