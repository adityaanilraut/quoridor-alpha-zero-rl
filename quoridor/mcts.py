"""PUCT Monte Carlo Tree Search guided by the policy/value network.

All nodes hold *canonical* states (player 0 to move), so values are always
from the perspective of the side to move and flip sign each ply.
"""

import math

import numpy as np

from .encoding import encode_state


class _Node:
    __slots__ = ("state", "is_expanded", "legal_mask", "value",
                 "child_priors", "child_N", "child_W", "child_Q", "children")

    def __init__(self, state):
        self.state = state
        self.is_expanded = False
        self.value = 0.0
        self.children = {}


class MCTS:
    def __init__(self, net, config):
        self.net = net
        self.sims = config.mcts_sims
        self.c_puct = config.c_puct
        self.alpha = config.dirichlet_alpha
        self.eps = config.dirichlet_eps
        # Gumbel MuZero search knobs (see run_gumbel).  getattr keeps old
        # configs / checkpoints that predate these fields working.
        self.n_considered = getattr(config, "gumbel_n_considered", 16)
        self.c_visit = getattr(config, "gumbel_c_visit", 50.0)
        self.c_scale = getattr(config, "gumbel_c_scale", 1.0)
        self._batch_queue = []  # (node, path) pairs for batched eval + backprop

    def run(self, canonical_state, add_noise=True):
        """Run search from a canonical state; return visit counts over actions."""
        root = _Node(canonical_state)
        self._expand(root)
        if add_noise:
            self._add_dirichlet(root)

        for _ in range(self.sims):
            node = root
            path = []
            while node.is_expanded and not node.state.is_terminal():
                a = self._select(node)
                path.append((node, a))
                if a not in node.children:
                    child_state = node.state.apply_action(a).canonical()
                    node.children[a] = _Node(child_state)
                node = node.children[a]

            if node.state.is_terminal():
                value = -1.0  # side to move at a terminal state has just lost
            else:
                value = self._expand(node)

            for parent, a in reversed(path):
                value = -value
                parent.child_N[a] += 1
                parent.child_W[a] += value
                parent.child_Q[a] = parent.child_W[a] / parent.child_N[a]

        return root.child_N.copy()

    def run_batched(self, canonical_state, add_noise=True):
        """Run search with batched network evaluation.

        Leaves are collected and evaluated in batches to amortise GPU launch
        overhead.  A pessimistic virtual loss is backpropped immediately to
        discourage other traversals from picking the same path; it is removed
        and replaced by the true value once the batch is flushed.
        """
        root = _Node(canonical_state)
        self._expand(root)
        if add_noise:
            self._add_dirichlet(root)

        for sim_idx in range(self.sims):
            node = root
            path = []
            while node.is_expanded and not node.state.is_terminal():
                a = self._select(node)
                path.append((node, a))
                if a not in node.children:
                    child_state = node.state.apply_action(a).canonical()
                    node.children[a] = _Node(child_state)
                node = node.children[a]

            if node.state.is_terminal():
                value = -1.0
                for parent, a in reversed(path):
                    value = -value
                    parent.child_N[a] += 1
                    parent.child_W[a] += value
                    parent.child_Q[a] = parent.child_W[a] / parent.child_N[a]
            else:
                # Virtual loss: treat the leaf as a loss for the side to move
                # there (value -1, sign-flipped up the path) so other in-batch
                # traversals avoid this line.  Removed in _flush_batch.
                vloss = -1.0
                for parent, a in reversed(path):
                    vloss = -vloss
                    parent.child_N[a] += 1
                    parent.child_W[a] += vloss
                    parent.child_Q[a] = parent.child_W[a] / parent.child_N[a]
                # Queue for batched eval; store path so we can backprop true value
                self._batch_queue.append((node, path))

            # Flush the batch every N leaves or at the end
            if len(self._batch_queue) >= 8:
                self._flush_batch()

        # Flush any remaining
        self._flush_batch()
        return root.child_N.copy()

    def _flush_batch(self):
        if not self._batch_queue:
            return
        items = self._batch_queue
        self._batch_queue = []
        nodes = [n for n, _ in items]
        paths = [p for _, p in items]
        states = [n.state for n in nodes]
        logits_batch, values_batch = self.net.predict_states(states)
        for node, path, logits, value in zip(nodes, paths, logits_batch, values_batch):
            # Expand the node with the real priors
            self._expand_from_logits(node, logits, value)
            # Replace the virtual-loss placeholder with the true value. The
            # visit counts (N) added during selection stay; only W is corrected
            # by removing the -/+1 placeholder and adding the real value.
            v = value.item()
            vloss = -1.0
            for parent, a in reversed(path):
                v = -v
                vloss = -vloss
                parent.child_W[a] += v - vloss
                parent.child_Q[a] = parent.child_W[a] / parent.child_N[a]

    def _expand(self, node):
        state = node.state
        mask = state.legal_actions_mask()
        logits, value = self.net.predict(encode_state(state))
        self._init_node(node, mask, logits, value)
        return value

    def _expand_from_logits(self, node, logits, value):
        mask = node.state.legal_actions_mask()
        self._init_node(node, mask, logits, value)

    def _init_node(self, node, mask, logits, value):
        logits = logits - logits.max()
        exp = np.exp(logits) * mask
        total = exp.sum()
        priors = exp / total if total > 0 else mask / max(mask.sum(), 1)

        a = mask.shape[0]
        node.legal_mask = mask
        # value may be a python float (single predict) or a 1-element array
        # (batched predict_states returns shape [L, 1]); .item() handles both.
        node.value = float(np.asarray(value).item())
        node.child_priors = priors.astype(np.float32)
        node.child_N = np.zeros(a, dtype=np.float32)
        node.child_W = np.zeros(a, dtype=np.float32)
        node.child_Q = np.zeros(a, dtype=np.float32)
        node.is_expanded = True

    def _select(self, node):
        sqrt_total = math.sqrt(node.child_N.sum() + 1.0)
        u = self.c_puct * node.child_priors * sqrt_total / (1.0 + node.child_N)
        scores = np.where(node.legal_mask > 0, node.child_Q + u, -1e9)
        return int(np.argmax(scores))

    def _add_dirichlet(self, node):
        idx = np.nonzero(node.legal_mask)[0]
        if idx.size == 0:
            return
        noise = np.random.dirichlet([self.alpha] * idx.size)
        node.child_priors[idx] = ((1 - self.eps) * node.child_priors[idx]
                                  + self.eps * noise)

    # ------------------------------------------------------------------ #
    # Gumbel MuZero search (EfficientZero V2 style)
    # ------------------------------------------------------------------ #
    def run_gumbel(self, canonical_state):
        """Gumbel-MuZero search on the true simulator.

        Returns ``(improved_policy, root_value)`` where

          * ``improved_policy`` is a probability vector over *all* actions:
            ``pi' = softmax(logits + sigma(completedQ))``.  It carries Gumbel's
            policy-improvement guarantee even at very low simulation counts, so
            it replaces the raw visit-count target.
          * ``root_value`` is the mixed root value estimate (search value),
            used for mixed value targets.

        No Dirichlet noise is needed: exploration comes from the Gumbel samples
        drawn at the root.
        """
        root = _Node(canonical_state)
        self._expand(root)
        legal = np.nonzero(root.legal_mask)[0]
        n_legal = int(legal.size)
        if n_legal == 0:
            return root.legal_mask.copy(), float(root.value)
        if n_legal == 1:
            pi = np.zeros_like(root.child_priors)
            pi[int(legal[0])] = 1.0
            return pi, float(root.value)

        # Sample m candidate actions at the root via Gumbel-Top-k.
        m = min(self.n_considered, n_legal)
        logits = np.log(np.clip(root.child_priors[legal], 1e-12, None))
        u = np.random.uniform(size=n_legal).clip(1e-12, 1.0)
        gumbel = -np.log(-np.log(u))
        order = np.argsort(-(gumbel + logits))[:m]
        candidates = [int(a) for a in legal[order]]
        g_by_a = {int(a): float(g) for a, g in zip(legal, gumbel)}
        l_by_a = {int(a): float(lg) for a, lg in zip(legal, logits)}

        # Sequential halving: spread the simulation budget across log2(m)
        # phases, halving the candidate set each phase by g + logit + sigma(Q).
        num_phases = max(1, int(math.ceil(math.log2(m))))
        budget = max(self.sims, m)  # try every candidate at least once
        while True:
            per = max(1, budget // (num_phases * len(candidates)))
            for _ in range(per):
                for a in candidates:
                    self._simulate(root, a)
            if len(candidates) <= 1:
                break
            sigma = (self.c_visit + float(root.child_N.max())) * self.c_scale
            candidates.sort(
                key=lambda a: g_by_a[a] + l_by_a[a] + sigma * (
                    float(root.child_Q[a]) if root.child_N[a] > 0
                    else float(root.value)),
                reverse=True,
            )
            candidates = candidates[:max(1, len(candidates) // 2)]

        return self._improved_policy(root), self._v_mix(root)

    def _simulate(self, root, root_action):
        """One simulation: force ``root_action`` at the root, then descend by
        the deterministic Gumbel rule to a leaf; expand and backprop."""
        node = root
        path = [(root, root_action)]
        if root_action not in node.children:
            node.children[root_action] = _Node(
                node.state.apply_action(root_action).canonical())
        node = node.children[root_action]
        while node.is_expanded and not node.state.is_terminal():
            a = self._select_gumbel(node)
            path.append((node, a))
            if a not in node.children:
                node.children[a] = _Node(node.state.apply_action(a).canonical())
            node = node.children[a]

        if node.state.is_terminal():
            value = -1.0  # side to move at a terminal state has just lost
        else:
            value = self._expand(node)

        for parent, a in reversed(path):
            value = -value
            parent.child_N[a] += 1
            parent.child_W[a] += value
            parent.child_Q[a] = parent.child_W[a] / parent.child_N[a]

    def _v_mix(self, node):
        """Value estimate mixing the network value with visited children's Q
        (Gumbel MuZero's v_mix), all from the node's side-to-move perspective."""
        sum_n = float(node.child_N.sum())
        if sum_n <= 0:
            return float(node.value)
        visited = node.child_N > 0
        sum_pi = float(node.child_priors[visited].sum())
        if sum_pi <= 1e-8:
            return float(node.value)
        weighted_q = float(
            (node.child_priors[visited] * node.child_Q[visited]).sum() / sum_pi)
        return (float(node.value) + sum_n * weighted_q) / (1.0 + sum_n)

    def _improved_policy(self, node):
        """pi' = softmax over legal actions of  log(prior) + sigma(completedQ),
        with completedQ = Q for visited actions and v_mix for unvisited ones."""
        legal = node.legal_mask > 0
        v_mix = self._v_mix(node)
        completed_q = np.where(node.child_N > 0, node.child_Q, v_mix)
        sigma = (self.c_visit + float(node.child_N.max())) * self.c_scale
        logits = np.full(node.child_priors.shape, -np.inf, dtype=np.float64)
        logits[legal] = np.log(np.clip(node.child_priors[legal], 1e-12, None))
        logits[legal] += sigma * completed_q[legal]
        return self._masked_softmax(logits, legal)

    @staticmethod
    def _masked_softmax(logits, legal):
        out = np.zeros(logits.shape, dtype=np.float32)
        if not legal.any():
            return out
        mx = logits[legal].max()
        exp = np.exp(logits[legal] - mx)
        out[legal] = (exp / exp.sum()).astype(np.float32)
        return out

    def _select_gumbel(self, node):
        """Deterministic interior selection: pick the legal action whose visit
        share most lags the improved policy (Gumbel MuZero's action rule)."""
        pi = self._improved_policy(node)
        sum_n = float(node.child_N.sum())
        score = pi - node.child_N / (1.0 + sum_n)
        score = np.where(node.legal_mask > 0, score, -np.inf)
        return int(np.argmax(score))
