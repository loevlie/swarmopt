"""Introspect a model to auto-generate a search space.

Supports PyTorch nn.Module and sklearn-compatible models (XGBoost, LightGBM, etc).
"""

import copy
import json
import re

from neuropt.search_space import Categorical, IntUniform, LogUniform, Uniform


# Activation types we know how to swap
ACTIVATION_TYPES = None  # lazy import to avoid hard torch dependency
ACTIVATION_MAP = {
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "leaky_relu": "LeakyReLU",
}


def _get_act_types():
    global ACTIVATION_TYPES
    if ACTIVATION_TYPES is None:
        import torch.nn as nn
        ACTIVATION_TYPES = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.Tanh)
    return ACTIVATION_TYPES


def _get_act_cls(name):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU}.get(name, nn.ReLU)


def introspect(model):
    """Walk a model's module tree and find tunable components.

    Returns a dict describing what was found: activation types, dropout
    layers and rates, batch norm presence, and module paths for each.
    """
    import torch.nn as nn
    act_types = _get_act_types()

    info = {
        "activations_found": set(),
        "activation_paths": [],
        "has_dropout": False,
        "dropout_rate": 0.0,
        "dropout_paths": [],
        "has_batchnorm": False,
        "batchnorm_paths": [],
        "has_conv": False,
        "has_linear": False,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    for name, mod in model.named_modules():
        if isinstance(mod, act_types):
            info["activations_found"].add(type(mod).__name__)
            info["activation_paths"].append(name)
        elif isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            info["has_dropout"] = True
            info["dropout_rate"] = mod.p
            info["dropout_paths"].append(name)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            info["has_batchnorm"] = True
            info["batchnorm_paths"].append(name)
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info["has_conv"] = True
        elif isinstance(mod, nn.Linear):
            info["has_linear"] = True

    return info


def build_search_space(info):
    """Generate a search space from introspection results."""
    space = {}

    if info["activation_paths"]:
        space["activation"] = Categorical(["relu", "gelu", "silu", "leaky_relu"])

    if info["has_dropout"]:
        space["dropout"] = Uniform(0.0, 0.5)

    if info["has_batchnorm"]:
        space["use_batchnorm"] = Categorical([True, False])

    # Always include training hyperparams
    space["lr"] = LogUniform(1e-4, 1e-1)
    space["wd"] = LogUniform(1e-6, 1e-2)
    space["optimizer"] = Categorical(["sgd", "adam", "adamw"])

    return space


def build_ml_context(info):
    """Generate LLM context describing what was found in the model."""
    parts = ["You are optimizing an existing PyTorch model.\n"]
    parts.append("## What was detected in the model\n")

    if info["activations_found"]:
        parts.append(f"- Activation functions: {', '.join(sorted(info['activations_found']))}")
        parts.append(f"  ({len(info['activation_paths'])} swappable activation layers)")
    if info["has_dropout"]:
        parts.append(f"- Dropout layers found (original rate: {info['dropout_rate']:.2f})")
    if info["has_batchnorm"]:
        parts.append(f"- Batch normalization: {len(info['batchnorm_paths'])} layers (can be toggled off)")
    if info["has_conv"]:
        parts.append("- Convolutional layers present")
    if info["has_linear"]:
        parts.append("- Fully connected layers present")
    parts.append(f"- Total parameters: {info['n_params']:,}")
    parts.append("")

    parts.append("## Guidance\n")
    parts.append(
        "- GELU and SiLU often outperform ReLU — try them\n"
        "- If overfitting: increase dropout, increase weight decay\n"
        "- If underfitting: reduce dropout (even to 0), reduce weight decay\n"
        "- BatchNorm usually helps but can sometimes hurt very small models\n"
        "- AdamW with lr ~1e-3 is a safe starting point\n"
        "- SGD needs higher lr (0.01-0.1) and is more sensitive to tuning\n"
        "- Read the training curves: train-val gap = overfitting, both stuck = underfitting\n"
        "- Balance exploration with exploitation — try new things but also refine what works"
    )
    return "\n".join(parts)


def apply_config(model, config, info):
    """Modify a deep-copied model in-place based on a config dict."""
    import torch.nn as nn

    # Swap activations
    if "activation" in config and info["activation_paths"]:
        act_cls = _get_act_cls(config["activation"])
        for path in info["activation_paths"]:
            _set_module(model, path, act_cls())

    # Set dropout rates
    if "dropout" in config:
        for path in info["dropout_paths"]:
            mod = _get_module(model, path)
            mod.p = config["dropout"]

    # Toggle batch norm off
    if "use_batchnorm" in config and not config["use_batchnorm"]:
        for path in info["batchnorm_paths"]:
            _set_module(model, path, nn.Identity())


def make_wrapped_train_fn(model, train_fn, info):
    """Create a train_fn that deep-copies the model and applies config modifications.

    The user's train_fn receives ``config["model"]`` (the modified model)
    plus any search space keys (activation, dropout, lr, wd, optimizer, etc).
    """
    def wrapped(config):
        modified = copy.deepcopy(model)
        apply_config(modified, config, info)
        config_with_model = dict(config)
        config_with_model["model"] = modified
        return train_fn(config_with_model)
    return wrapped


# ── Module path helpers ──────────────────────────────────────────────────

def _get_module(model, path):
    """Get a module by dot-separated path (handles Sequential integer indices)."""
    parts = path.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _set_module(model, path, new_module):
    """Replace a module at the given dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


# ── Sklearn-compatible model introspection ───────────────────────────────

SKIP_PARAMS = {
    "random_state", "seed", "n_jobs", "verbose", "verbosity", "silent",
    "objective", "eval_metric", "use_label_encoder", "device", "gpu_id",
    "tree_method", "predictor", "booster", "importance_type", "callbacks",
    "enable_categorical", "feature_types", "max_cat_to_onehot",
    "max_cat_threshold", "interaction_constraints", "monotone_constraints",
    "base_score", "validate_parameters", "nthread",
}


def is_sklearn_compatible(model):
    """Check if a model has sklearn-style get_params/set_params."""
    return hasattr(model, "get_params") and hasattr(model, "set_params")


def introspect_sklearn(model):
    """Introspect an sklearn-compatible model (XGBoost, LightGBM, sklearn, etc).

    Returns a dict with model class name and tunable parameters + current values.
    Includes None-valued params since many libraries use None as "use internal default."
    """
    params = model.get_params()
    tunable = {}
    for name, value in params.items():
        if name in SKIP_PARAMS:
            continue
        if isinstance(value, (int, float, bool, str)):
            tunable[name] = value
        elif value is None:
            tunable[name] = None  # include — LLM knows the real defaults

    return {
        "model_type": type(model).__name__,
        "model_module": type(model).__module__,
        "all_params": params,
        "tunable_params": tunable,
    }


def build_sklearn_search_space_with_llm(info, backend):
    """Ask the LLM for reasonable search ranges given the model type and params."""
    model_type = info["model_type"]
    tunable = info["tunable_params"]

    prompt = (
        f"You are setting up a hyperparameter search for a {model_type} model.\n\n"
        f"Here are its tunable parameters and current values:\n"
    )
    for name, value in tunable.items():
        prompt += f"  {name} = {value!r} ({type(value).__name__})\n"

    prompt += (
        f"\nFor each parameter, provide a search range as JSON. Use this format:\n"
        f'{{"param_name": {{"type": "int"|"float"|"log_float"|"bool"|"choice", '
        f'"min": ..., "max": ..., "choices": [...]}}}}\n\n'
        f"Rules:\n"
        f"- Only include parameters worth tuning (skip ones that rarely matter)\n"
        f"- Use \"log_float\" for parameters that span orders of magnitude (like learning_rate, reg_alpha, reg_lambda)\n"
        f"- Use \"int\" for integer parameters with a range\n"
        f"- Use \"float\" for bounded float parameters\n"
        f"- Use \"bool\" for boolean toggles\n"
        f"- Use \"choice\" for categorical options\n"
        f"- Choose ranges that a practitioner would actually search over\n\n"
        f"Respond with ONLY the JSON object. No explanation."
    )

    response = backend.generate(prompt, max_tokens=1024)

    # Parse response
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        return _fallback_sklearn_search_space(info)

    try:
        ranges = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback_sklearn_search_space(info)

    space = {}
    for name, spec in ranges.items():
        if name not in tunable:
            continue
        try:
            t = spec.get("type", "float")
            if t == "log_float":
                lo, hi = float(spec["min"]), float(spec["max"])
                if lo <= 0:
                    lo = 1e-8
                if hi <= lo:
                    hi = lo * 100
                space[name] = LogUniform(lo, hi)
            elif t == "float":
                space[name] = Uniform(float(spec["min"]), float(spec["max"]))
            elif t == "int":
                space[name] = IntUniform(int(spec["min"]), int(spec["max"]))
            elif t == "bool":
                space[name] = Categorical([True, False])
            elif t == "choice":
                space[name] = Categorical(spec["choices"])
        except (KeyError, TypeError, ValueError):
            continue

    if not space:
        return _fallback_sklearn_search_space(info)

    return space


_KNOWN_RANGES = {
    "max_depth":        IntUniform(3, 12),
    "n_estimators":     IntUniform(50, 500),
    "learning_rate":    LogUniform(1e-3, 0.3),
    "eta":              LogUniform(1e-3, 0.3),
    "min_child_weight": IntUniform(1, 10),
    "subsample":        Uniform(0.5, 1.0),
    "colsample_bytree": Uniform(0.5, 1.0),
    "colsample_bylevel": Uniform(0.5, 1.0),
    "colsample_bynode": Uniform(0.5, 1.0),
    "gamma":            LogUniform(1e-5, 10.0),
    "reg_alpha":        LogUniform(1e-5, 10.0),
    "reg_lambda":       LogUniform(1e-5, 10.0),
    "max_delta_step":   IntUniform(0, 10),
    "scale_pos_weight": Uniform(0.5, 5.0),
    "max_leaves":       IntUniform(0, 128),
    "num_leaves":       IntUniform(15, 127),
    "min_data_in_leaf": IntUniform(5, 100),
    "bagging_fraction": Uniform(0.5, 1.0),
    "feature_fraction": Uniform(0.5, 1.0),
    "max_features":     Uniform(0.5, 1.0),
    "min_samples_split": IntUniform(2, 20),
    "min_samples_leaf": IntUniform(1, 20),
    "max_samples":      Uniform(0.5, 1.0),
}


def _fallback_sklearn_search_space(info):
    """Heuristic-based search space when LLM isn't available or fails."""
    space = {}
    for name, value in info["tunable_params"].items():
        if name in _KNOWN_RANGES:
            space[name] = _KNOWN_RANGES[name]
        elif isinstance(value, bool):
            space[name] = Categorical([True, False])
        elif isinstance(value, int) and value > 0:
            space[name] = IntUniform(max(1, value // 3), value * 3)
        elif isinstance(value, float) and value > 0:
            if value < 0.01 or "reg" in name or "alpha" in name or "lambda" in name:
                space[name] = LogUniform(value / 10, min(value * 10, 100))
            elif value <= 1.0:
                space[name] = Uniform(max(0.0, value - 0.3), min(1.0, value + 0.3))
            else:
                space[name] = Uniform(value / 3, value * 3)
    return space


def build_sklearn_ml_context(info, space):
    """Generate LLM context for an sklearn-compatible model."""
    parts = [f"You are optimizing a {info['model_type']} model.\n"]
    parts.append("## Parameters being searched\n")
    for name, dim in space.items():
        current = info["tunable_params"].get(name, "?")
        parts.append(f"- {name} (current: {current!r}): {dim}")
    parts.append("")
    parts.append(
        "## Guidance\n"
        "- Read the training curves to spot overfitting vs underfitting\n"
        "- If overfitting: increase regularization, reduce model complexity\n"
        "- If underfitting: reduce regularization, increase complexity\n"
        "- Balance exploration with exploitation\n"
        "- Don't repeat configs that have already been tried"
    )
    return "\n".join(parts)


def make_sklearn_wrapped_train_fn(model, train_fn):
    """Wrap train_fn to clone the model and set params from config each call."""
    def wrapped(config):
        # Separate search params from non-model keys
        model_params = {}
        extra = {}
        param_names = set(model.get_params().keys())
        for k, v in config.items():
            if k in param_names:
                model_params[k] = v
            else:
                extra[k] = v

        from sklearn.base import clone
        cloned = clone(model)
        cloned.set_params(**model_params)

        config_with_model = dict(config)
        config_with_model["model"] = cloned
        return train_fn(config_with_model)

    return wrapped
