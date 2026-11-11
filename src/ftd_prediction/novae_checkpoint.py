"""Offline, explicitly pinned expression-only Novae checkpoint compatibility.

The default loader remains strict. No third-party files are patched, no model is
downloaded, and compatibility never removes or substitutes an active tensor.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path


COMPATIBILITY = "novae-human-0-expression-v1"
REPOSITORY = "prism-oncology/novae-human-0"
REVISION = "b8c0a5d7612bac6bc719ab57ed3cd16ad814728c"
NOVAE_VERSION = "1.1.3"
HISTO_EMBEDDINGS = "histo_embeddings"  # novae._constants.Keys.HISTO_EMBEDDINGS
PINNED_FILES = {
    "config.json": {"sha256": "71070bb73a5f252ad2394a1b7132ff92b2a3833e6f4420640aa8a48ffc39a62f",
                    "size_bytes": 1025622},
    "model.safetensors": {"sha256": "7d1e4a051469d14e796994faa34a41556183e78757f5dc70a99dbd9f6a19ce87",
                          "size_bytes": 128175588},
}
INACTIVE_SHAPES = {
    "encoder.mlp_fusion.0.weight": (114, 114),
    "encoder.mlp_fusion.0.bias": (114,),
    "encoder.mlp_fusion.2.weight": (64, 114),
    "encoder.mlp_fusion.2.bias": (64,),
    "encoder.mlp_fusion.4.weight": (64, 64),
    "encoder.mlp_fusion.4.bias": (64,),
}
# All 78 public tensors plus the six approved F32 tensors, sorted by name.
STATE_SCHEMA_SHA256 = "03eaa734c576daed344ad51e5af7efc82de46906941d575463515054e23fb8c6"
_CONSTRUCTOR_DEFAULTS = {
    "adata": None, "embedding_size": 100, "embedding_name": None,
    "min_prototypes_ratio": 0.3, "n_hops_local": 2, "n_hops_view": 2,
    "temperature": 0.1, "output_size": 64, "heads": 8, "hidden_size": 128,
    "num_layers": 10, "batch_size": 256, "num_prototypes": 512,
    "panel_subset_size": 0.8, "background_noise_lambda": 8.0,
    "sensitivity_noise_std": 0.05, "dropout_rate": 0.0,
    "histo_embedding_size": 50, "pca_init_reference_index": None,
    "scgpt_model_dir": None, "var_names": None, "mode_kwargs": None,
}
_RATIONALE = (
    "The pinned public expression checkpoint predates six image-fusion tensors. "
    "Novae 1.1.3 GraphEncoder.forward uses them only when histo_embeddings is "
    "present; this compatibility path rejects that input. Only these inactive "
    "tensors are initialized to zero; all original tensors are copied unchanged."
)


def validate_compatibility(mode):
    if not isinstance(mode, str) or mode not in ("strict", COMPATIBILITY):
        raise ValueError("Unsupported checkpoint_compatibility mode")
    return mode


def reject_image_input(adata):
    """Reject the actual official AnnData H&E key before an adapter drops it."""
    if isinstance(adata, (list, tuple)):
        for item in adata:
            reject_image_input(item)
    elif adata is not None and HISTO_EMBEDDINGS in adata.obsm:
        raise ValueError("Expression checkpoint compatibility rejects histo_embeddings (H&E input)")


def _reject_encoder_images(module, args, kwargs):
    data = args[0] if args else kwargs.get("data")
    if data is not None and hasattr(data, HISTO_EMBEDDINGS):
        raise ValueError("Expression checkpoint compatibility rejects histo_embeddings (H&E input)")


def _validate_runtime(novae):
    if novae.__version__ != NOVAE_VERSION:
        raise ValueError("Checkpoint compatibility requires Novae 1.1.3")
    parameters = dict(inspect.signature(novae.Novae.__init__).parameters)
    parameters.pop("self", None)
    if (list(parameters) != list(_CONSTRUCTOR_DEFAULTS)
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or p.default != _CONSTRUCTOR_DEFAULTS[name]
                   for name, p in parameters.items())):
        raise ValueError("Unexpected Novae 1.1.3 constructor contract")


def _read_pinned_file(path, identity):
    """Hash the very bytes that will be parsed; cap reads at the pinned size."""
    if not path.is_file() or path.stat().st_size != identity["size_bytes"]:
        raise ValueError("Public checkpoint file size does not match its pin")
    with path.open("rb") as handle:
        content = handle.read(identity["size_bytes"] + 1)
    if (len(content) != identity["size_bytes"]
            or hashlib.sha256(content).hexdigest() != identity["sha256"]):
        raise ValueError("Public checkpoint SHA256 does not match its pin")
    return content


def _schema_hash(state):
    import torch
    if any(t.dtype != torch.float32 for t in state.values()):
        raise ValueError("Unexpected current Novae state dtype")
    schema = {name: {"shape": list(t.shape), "dtype": "F32"} for name, t in state.items()}
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _complete_expression_state(expected, original):
    """Validate every tensor before synthesizing exactly the inactive allowlist."""
    import torch
    if set(original) - set(expected):
        raise ValueError("Unexpected checkpoint tensor keys")
    if set(expected) - set(original) != set(INACTIVE_SHAPES):
        raise ValueError("Checkpoint must lack exactly the six inactive fusion tensors")
    for name, tensor in original.items():
        target = expected[name]
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError("Checkpoint active tensor shape/dtype mismatch")
    for name, shape in INACTIVE_SHAPES.items():
        if tuple(expected[name].shape) != shape or expected[name].dtype != torch.float32:
            raise ValueError("Unexpected inactive fusion tensor shape/dtype")
    complete = dict(original)
    complete.update({name: torch.zeros_like(expected[name]) for name in INACTIVE_SHAPES})
    return complete


def load_expression_checkpoint(directory, compatibility="strict", *, adata=None):
    """Return (model, receipt); only the exact opt-in path returns a receipt.

    Export with normal ``save_pretrained``. The resulting complete model reloads
    through the ordinary strict path, without applying compatibility a second time.
    """
    validate_compatibility(compatibility)
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("A local checkpoint directory is required")
    if compatibility == COMPATIBILITY:
        reject_image_input(adata)
    import novae
    if novae.__version__ != NOVAE_VERSION:
        raise ValueError("This loader requires Novae 1.1.3")
    if compatibility == "strict":
        return novae.Novae.from_pretrained(directory, local_files_only=True,
                                            map_location="cpu", strict=True), None
    _validate_runtime(novae)
    # Authenticate both files before parsing or constructing a model. No mutable
    # path is reopened to load unchecked weight bytes after verification.
    config_bytes = _read_pinned_file(directory / "config.json", PINNED_FILES["config.json"])
    weight_bytes = _read_pinned_file(directory / "model.safetensors", PINNED_FILES["model.safetensors"])
    config = json.loads(config_bytes)
    if (config.get("embedding_name") is not None or config.get("scgpt_model_dir") is not None
            or config.get("mode_kwargs", {}).get("multimodal", False)):
        raise ValueError("Compatibility requires the pinned expression constructor")
    model = novae.Novae(**config)
    if (model.cell_embedder is None or model.hparams.embedding_name is not None
            or model.mode.multimodal):
        raise ValueError("Compatibility requires an expression model")
    expected = model.state_dict()
    if _schema_hash(expected) != STATE_SCHEMA_SHA256:
        raise ValueError("Unexpected current Novae 1.1.3 state schema")
    import torch
    from safetensors.torch import load
    original = load(weight_bytes)
    del weight_bytes
    complete = _complete_expression_state(expected, original)
    model.load_state_dict(complete, strict=True)
    current = model.state_dict()
    if any(not torch.equal(current[name].view(torch.uint8), tensor.view(torch.uint8))
           for name, tensor in original.items()):
        raise ValueError("Active checkpoint tensor bytes changed during strict load")
    model.eval()
    # Mirror the official from_pretrained post-load behavior, including freezing
    # the gene embedding; the adapter controls frozen/fine-tune fitting as before.
    model.mode.as_pretrained()
    model._model_name = str(directory)
    model.cell_embedder.embedding.weight.requires_grad_(False)
    model.encoder.register_forward_pre_hook(_reject_encoder_images, with_kwargs=True)
    receipt = {
        "mode": COMPATIBILITY, "repository": REPOSITORY, "revision": REVISION,
        "novae_version": NOVAE_VERSION, "original_files": {k: dict(v) for k, v in PINNED_FILES.items()},
        "current_state_schema_sha256": STATE_SCHEMA_SHA256,
        "active_tensor_count": len(original), "active_tensor_bytes_unchanged": True,
        "verification_stage": "initial_checkpoint_load_before_any_fitting",
        "strict_load_state_dict": True, "image_input_key_rejected": HISTO_EMBEDDINGS,
        "synthesized_inactive_tensors": [
            {"name": name, "shape": list(shape), "dtype": "float32", "initialization": "zeros"}
            for name, shape in INACTIVE_SHAPES.items()],
        "rationale": _RATIONALE,
        "upstream_evidence": {
            "commit": "8f9374ad9fdb708babd642843312cb876957cc36",
            "paths": ["tests/test_model.py:270-285", "novae/module/encode.py:44-50,64-65"],
        },
        "loader_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return model, receipt
