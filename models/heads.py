"""
Task heads, losses, and metrics for GEDI-ORDER.

This module lets one backbone feed several scientific "tasks" without
rewriting the network each time:

    - 'classification'     : N-way softmax (alive/dead or N cell states)
    - 'regression'         : continuous readout(s)
    - 'survival_cox'       : DeepSurv-style single risk score, trained with the
                             Cox partial likelihood -> "time to event"
    - 'survival_discrete'  : logistic-hazard (nnet-survival) over K time bins,
                             which also supports right-censoring naturally and
                             is easy to extend to competing/multiple events

The survival heads are what turn this from a "does the cell die" classifier
into a "when does the event happen" predictor.

References
----------
Cox partial likelihood / DeepSurv: Katzman et al. 2018, BMC Med. Res. Methodol.
Logistic-hazard discrete survival: Gensheimer & Narasimhan 2019, PeerJ
Concordance index: Harrell et al. 1982, JAMA
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Tasks that produce a single scalar per image and can be explained as one
# "risk"/"value" channel by the XAI engine.
SCALAR_TASKS = ("regression", "survival_cox")
SURVIVAL_TASKS = ("survival_cox", "survival_discrete")
ALL_TASKS = ("classification", "regression", "survival_cox", "survival_discrete")


# --------------------------------------------------------------------------- #
# Head construction
# --------------------------------------------------------------------------- #
def build_head(features: tf.Tensor, task: str, num_outputs: int,
               hidden: Tuple[int, ...] = (256,), dropout: float = 0.3,
               name: str = "head") -> tf.Tensor:
    """Attach a task-specific head to a pooled feature vector.

    Args:
        features: 2-D tensor (batch, feat_dim) from a global-pooled backbone.
        task: one of ALL_TASKS.
        num_outputs: classes (classification), targets (regression),
            or time bins (survival_discrete). Ignored for survival_cox (=1).
        hidden: sizes of shared dense layers before the output.
        dropout: dropout rate between hidden layers (0 disables).
        name: prefix for layer names (also used as the output layer name so
            the XAI engine and losses can find it).

    Returns:
        Output tensor. Grad-CAM should target a conv layer, not this.
    """
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {ALL_TASKS}")

    x = features
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation="relu", name=f"{name}_fc{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{name}_drop{i}")(x)

    if task == "classification":
        return layers.Dense(num_outputs, activation="softmax", name=name)(x)
    if task == "regression":
        return layers.Dense(num_outputs, activation="linear", name=name)(x)
    if task == "survival_cox":
        # Single log-risk score; higher => event sooner. No activation.
        return layers.Dense(1, activation="linear", name=name)(x)
    # survival_discrete: one hazard logit per time bin (sigmoid applied in loss).
    return layers.Dense(num_outputs, activation="linear", name=name)(x)


# --------------------------------------------------------------------------- #
# Cox partial likelihood (DeepSurv)
# --------------------------------------------------------------------------- #
def cox_partial_likelihood_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), computed per batch.

    Args:
        y_true: (batch, 2) float tensor of [event_time, event_indicator],
            event_indicator = 1 if the event was observed, 0 if right-censored.
        y_pred: (batch, 1) log-risk scores from a survival_cox head.

    Returns:
        Scalar loss. Larger risk for shorter survival is rewarded. The risk
        set for each sample is approximated by the current batch, so use a
        reasonably large batch size for a stable estimate.
    """
    y_true = tf.cast(y_true, tf.float32)
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    # risk_set[i, j] = 1 if sample j is still at risk when i has its event
    # (i.e. time_j >= time_i).
    at_risk = tf.cast(time[None, :] >= time[:, None], tf.float32)

    # log sum_{j in risk set of i} exp(risk_j), done stably with masking.
    neg_inf = tf.fill(tf.shape(at_risk), tf.constant(-1e9, tf.float32))
    masked_risk = tf.where(at_risk > 0, risk[None, :] + tf.zeros_like(at_risk), neg_inf)
    log_cumsum = tf.reduce_logsumexp(masked_risk, axis=1)

    # Only observed events contribute to the partial likelihood.
    loglik = event * (risk - log_cumsum)
    n_events = tf.reduce_sum(event) + tf.keras.backend.epsilon()
    return -tf.reduce_sum(loglik) / n_events


# --------------------------------------------------------------------------- #
# Discrete-time logistic hazard (nnet-survival)
# --------------------------------------------------------------------------- #
def make_survival_array(time_bin: np.ndarray, event: np.ndarray,
                        num_bins: int) -> np.ndarray:
    """Encode (time_bin, event) into the (batch, 2*num_bins) target for
    `discrete_time_hazard_loss`.

    First `num_bins` columns: 1 while the subject is event-free through bin k.
    Last  `num_bins` columns: 1 in the single bin where an observed event
    occurred (all zero for censored subjects).

    Args:
        time_bin: int array (batch,), index in [0, num_bins) of the last bin
            the subject was observed in.
        event: int/bool array (batch,), 1 if event observed, 0 if censored.
        num_bins: number of discrete time intervals.
    """
    time_bin = np.asarray(time_bin).astype(int)
    event = np.asarray(event).astype(int)
    n = len(time_bin)
    surv = np.zeros((n, num_bins), dtype=np.float32)
    hit = np.zeros((n, num_bins), dtype=np.float32)
    for i in range(n):
        t = min(max(time_bin[i], 0), num_bins - 1)
        # Survived (event-free) through every bin up to and including t.
        surv[i, : t + 1] = 1.0
        if event[i]:
            hit[i, t] = 1.0
    return np.concatenate([surv, hit], axis=1)


def discrete_time_hazard_loss(num_bins: int):
    """Return a Keras loss for logistic-hazard discrete-time survival.

    Args:
        num_bins: number of time intervals K. y_true is (batch, 2K) from
            `make_survival_array`; y_pred is (batch, K) hazard logits.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        surv = y_true[:, :num_bins]      # event-free-through-bin mask
        hit = y_true[:, num_bins:]       # event-in-bin one-hot
        hazard = tf.sigmoid(tf.cast(y_pred, tf.float32))
        eps = tf.keras.backend.epsilon()
        # Likelihood: survive the bins passed through, then (if event) fail in
        # the event bin. Contribution per bin:
        #   event-free bin:            log(1 - hazard)
        #   event bin (event==1):      log(hazard)
        survived = surv * (1.0 - hit)    # bins passed through with no event
        ll = survived * tf.math.log(1.0 - hazard + eps) + hit * tf.math.log(hazard + eps)
        return -tf.reduce_mean(tf.reduce_sum(ll, axis=1))
    loss.__name__ = "discrete_time_hazard_loss"
    return loss


def discrete_survival_to_risk(hazard_logits: np.ndarray) -> np.ndarray:
    """Collapse per-bin hazard logits into a single scalar risk for ranking
    (used by the concordance index and by the XAI engine).

    Risk = num_bins - E[survival bins] = expected number of bins failed,
    so higher = event sooner.
    """
    hazard = 1.0 / (1.0 + np.exp(-np.asarray(hazard_logits, dtype=np.float64)))
    surv = np.cumprod(1.0 - hazard, axis=1)      # S(k) survival curve
    expected_lifetime = surv.sum(axis=1)         # sum of survival probs
    return -expected_lifetime                    # higher risk => shorter life


# --------------------------------------------------------------------------- #
# Concordance index (Harrell's C) — evaluation metric for survival
# --------------------------------------------------------------------------- #
def concordance_index(event_time: np.ndarray, risk: np.ndarray,
                      event_observed: np.ndarray) -> float:
    """Harrell's concordance index for right-censored data.

    A pair (i, j) is comparable if the one with the shorter time had an
    observed event. It is concordant if that subject has the higher risk.

    Args:
        event_time: (n,) observed times.
        risk: (n,) predicted risk scores (higher => event sooner).
        event_observed: (n,) 1 if event observed, 0 if censored.

    Returns:
        C-index in [0, 1]; 0.5 is random, 1.0 is perfect ranking. Returns
        float('nan') if there are no comparable pairs.
    """
    event_time = np.asarray(event_time, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    event_observed = np.asarray(event_observed).astype(bool)

    n = len(event_time)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if event_time[j] > event_time[i]:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if permissible == 0:
        return float("nan")
    return concordant / permissible
