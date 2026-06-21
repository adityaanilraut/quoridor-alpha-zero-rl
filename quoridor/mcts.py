"""PUCT Monte Carlo Tree Search guided by the policy/value network.

All nodes hold *canonical* states (player 0 to move), so values are always
from the perspective of the side to move and flip sign each ply.
"""

import math

import numpy as np

from .encoding import encode_state


class _Node:
    __slots__ = ("state", "is_expanded", "legal_mask",
                 "child_priors", "child_N", "child_W", "child_Q", "children")

    def __init__(self, state):
        self.state = state
        self.is_expanded = False
        self.children = {}


class MCTS:
    def __init__(self, net, config):
        self.net = net
        self.sims = config.mcts_sims
        self.c_puct = config.c_puct
        self.alpha = config.dirichlet_alpha
        self.eps = config.dirichlet_eps
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
