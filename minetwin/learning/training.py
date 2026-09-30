import copy
import json
import random
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from minetwin.learning.contracts import TrainingConfig
from minetwin.learning.data import CachedSplit
from minetwin.learning.metrics import classification_metrics
from minetwin.learning.model import build_risk_mlp, require_torch
from minetwin.learning.scaling import FederatedStandardScaler


def fit_centralized_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, list[dict]]:
    torch = require_torch()
    _seed(config.seed, torch)
    architecture = (train_features.shape[1], config.hidden_sizes, 5)
    model = _new_model(architecture, config.seed, torch)
    history = _fit(
        model,
        train_features,
        train_labels,
        config.epochs,
        config,
        _class_weights(train_labels, torch),
        config.seed,
    )
    probabilities = _predict(
        model, validation_features, config.batch_size, torch
    )
    return probabilities, history


def train_learning_regimes(
    cache: Path,
    output: Path,
    config: TrainingConfig | None = None,
) -> dict:
    effective = config or TrainingConfig()
    torch = require_torch()
    _seed(effective.seed, torch)
    cache = Path(cache)
    train = CachedSplit.load(cache / "train.npz")
    validation = CachedSplit.load(cache / "validation.npz")
    if train.feature_names != validation.feature_names:
        raise ValueError("Entrenamiento y validacion no comparten variables.")
    scaler = FederatedStandardScaler.fit(train.features, train.nodes)
    train_features = scaler.transform(train.features)
    validation_features = scaler.transform(validation.features)
    class_weights = _class_weights(train.labels, torch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        scaler.save(output / "scaler.npz")
        architecture = (train_features.shape[1], effective.hidden_sizes, 5)
        results = {}
        training_history = {}
        local_models = {}
        local_history = {}
        for index, node in enumerate(np.unique(train.nodes)):
            selected = train.nodes == node
            model = _new_model(architecture, effective.seed, torch)
            local_history[str(node)] = _fit(
                model,
                train_features[selected],
                train.labels[selected],
                effective.epochs,
                effective,
                class_weights,
                effective.seed + index,
            )
            local_models[str(node)] = model
            _save_model(model, output / f"local_{node}.pt", architecture, torch)
        training_history["local"] = local_history
        local_probabilities = np.empty((len(validation.labels), 5), dtype=np.float32)
        for node, model in local_models.items():
            selected = validation.nodes == node
            local_probabilities[selected] = _predict(
                model, validation_features[selected], effective.batch_size, torch
            )
        results["local"] = _grouped_metrics(
            validation.labels, local_probabilities, validation.nodes
        )
        centralized = _new_model(architecture, effective.seed, torch)
        training_history["centralized"] = _fit(
            centralized,
            train_features,
            train.labels,
            effective.epochs,
            effective,
            class_weights,
            effective.seed,
        )
        _save_model(
            centralized, output / "centralized.pt", architecture, torch
        )
        results["centralized"] = _evaluate(
            centralized,
            validation_features,
            validation.labels,
            validation.nodes,
            effective.batch_size,
            torch,
        )
        for name, proximal_mu in (("fedavg", 0.0), ("fedprox", effective.proximal_mu)):
            model, history = _federated_fit(
                train_features,
                train.labels,
                train.nodes,
                architecture,
                effective,
                class_weights,
                proximal_mu,
                torch,
            )
            training_history[name] = history
            _save_model(model, output / f"{name}.pt", architecture, torch)
            results[name] = _evaluate(
                model,
                validation_features,
                validation.labels,
                validation.nodes,
                effective.batch_size,
                torch,
            )
        report = {
            "config": asdict(effective),
            "torch": version("torch"),
            "features": train_features.shape[1],
            "training_examples": len(train.labels),
            "validation_examples": len(validation.labels),
            "training_history": training_history,
            "results": results,
        }
        (output / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    except (Exception, KeyboardInterrupt):
        shutil.rmtree(output)
        raise


def _federated_fit(
    features,
    labels,
    nodes,
    architecture,
    config,
    class_weights,
    proximal_mu,
    torch,
):
    global_model = _new_model(architecture, config.seed, torch)
    node_ids = np.unique(nodes)
    history = []
    for round_index in range(config.federated_rounds):
        reference = copy.deepcopy(global_model.state_dict())
        states = []
        sizes = []
        losses = []
        for node_index, node in enumerate(node_ids):
            selected = nodes == node
            local = _new_model(architecture, config.seed, torch)
            local.load_state_dict(reference)
            local_history = _fit(
                local,
                features[selected],
                labels[selected],
                config.local_epochs,
                config,
                class_weights,
                config.seed + round_index * len(node_ids) + node_index,
                reference if proximal_mu else None,
                proximal_mu,
            )
            states.append(local.state_dict())
            size = int(np.sum(selected))
            sizes.append(size)
            losses.append(local_history[-1]["training_loss"] * size)
        global_model.load_state_dict(_weighted_state(states, sizes, torch))
        history.append(
            {
                "round": round_index + 1,
                "training_loss": sum(losses) / sum(sizes),
            }
        )
    return global_model, history


def _fit(
    model,
    features,
    labels,
    epochs,
    config,
    class_weights,
    seed,
    reference=None,
    proximal_mu=0.0,
):
    torch = require_torch()
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.0
        for start in range(0, len(x), config.batch_size):
            indexes = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[indexes]), y[indexes])
            if reference is not None:
                proximal = sum(
                    torch.sum((parameter - reference[name]) ** 2)
                    for name, parameter in model.named_parameters()
                )
                loss = loss + proximal_mu * proximal / 2
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indexes)
        history.append({"epoch": epoch + 1, "training_loss": total_loss / len(x)})
    return history


def _weighted_state(states, sizes, torch):
    total = sum(sizes)
    return {
        name: sum(
            state[name] * (size / total)
            for state, size in zip(states, sizes, strict=True)
        )
        for name in states[0]
    }


def _predict(model, features, batch_size, torch):
    model.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size])
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(batches)


def _evaluate(model, features, labels, nodes, batch_size, torch):
    probabilities = _predict(model, features, batch_size, torch)
    return _grouped_metrics(labels, probabilities, nodes)


def _grouped_metrics(labels, probabilities, nodes):
    return {
        "global": classification_metrics(labels, probabilities),
        "nodes": {
            str(node): classification_metrics(
                labels[nodes == node], probabilities[nodes == node]
            )
            for node in np.unique(nodes)
        },
    }


def _class_weights(labels, torch):
    counts = np.bincount(labels, minlength=5)
    if np.any(counts == 0):
        raise ValueError("Entrenamiento requiere ejemplos de las cinco clases.")
    weights = np.sqrt(len(labels) / (5 * counts))
    return torch.from_numpy(weights.astype(np.float32))


def _save_model(model, path, architecture, torch):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": architecture[0],
            "hidden_sizes": architecture[1],
            "classes": architecture[2],
        },
        path,
    )


def _new_model(architecture, seed, torch):
    torch.manual_seed(seed)
    return build_risk_mlp(*architecture)


def _seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
