"""Small C-SWM-style slot encoder and action-conditioned transition models.

The encoder follows the official C-SWM small CNN/MLP path. Predictor variants
share the same [B, objects, state_dim] slots and action routing convention.
This is a controlled pilot implementation, not a claim of exact paper settings.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


def _get(config, key, default):
    if not hasattr(config, "get"):
        return getattr(config, key, default)
    if key in config:
        return config[key]
    # Accept either a compact model config or the frozen protocol document.
    paths = {
        "image_shape": (("environment", "image_shape_chw"),),
        "num_objects": (("representation", "num_slots"), ("environment", "objects")),
        "state_dim": (("representation", "slot_dim"),),
        "hidden": (("representation", "hidden_dim"),
                   ("stage_a_encoder_fit", "gnn_hidden_dim"),
                   ("stage_b_predictor_fit", "transformer6", "d_model")),
        "action_dim": (("stage_a_encoder_fit", "action_dim_per_object"),),
    }
    for path in paths.get(key, ()):
        value = config
        for part in path:
            value = value.get(part) if hasattr(value, "get") else None
            if value is None:
                break
        if value is not None:
            return value
    return default


@contextmanager
def _seeded(seed):
    # Keep module initialization from advancing the experiment's minibatch RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def encode_actions(actions, num_objects=5, action_dim=4):
    """Decode integer object*direction actions into per-slot one-hot vectors."""
    actions = actions.reshape(-1).long()
    target = torch.div(actions, action_dim, rounding_mode="floor")
    direction = F.one_hot(actions.remainder(action_dim), action_dim).to(torch.float32)
    slots = torch.arange(num_objects, device=actions.device).view(1, num_objects)
    selected = target.unsqueeze(1).eq(slots).unsqueeze(-1)
    return direction.unsqueeze(1) * selected.to(direction.dtype)


class SlotEncoder(nn.Module):
    """C-SWM small encoder: sigmoid object maps, then a shared object MLP."""
    def __init__(self, config):
        super().__init__()
        channels = int(_get(config, "image_shape", [3, 50, 50])[0])
        objects = int(_get(config, "num_objects", 5))
        state_dim = int(_get(config, "state_dim", 16))
        hidden = int(_get(config, "hidden", 128))
        cnn_hidden = hidden // 16
        self.num_objects = objects
        self.state_dim = state_dim
        self.cnn1 = nn.Conv2d(channels, cnn_hidden, kernel_size=10, stride=10)
        self.bn1 = nn.BatchNorm2d(cnn_hidden)
        self.cnn2 = nn.Conv2d(cnn_hidden, objects, kernel_size=1)
        self.object_fc1 = nn.Linear(25, hidden)
        self.object_fc2 = nn.Linear(hidden, hidden)
        self.object_ln = nn.LayerNorm(hidden)
        self.object_fc3 = nn.Linear(hidden, state_dim)

    def forward(self, image):
        maps = torch.sigmoid(self.cnn2(F.relu(self.bn1(self.cnn1(image)))))
        objects = maps.flatten(2)
        h = F.relu(self.object_fc1(objects))
        h = F.relu(self.object_ln(self.object_fc2(h)))
        return self.object_fc3(h)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, state_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, state_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class LocalPredictor(nn.Module):
    """Shared per-slot residual function with optional local/global summary."""
    def __init__(self, config, message_dim, message_kind):
        super().__init__()
        self.num_objects = int(_get(config, "num_objects", 5))
        self.state_dim = int(_get(config, "state_dim", 16))
        self.action_dim = int(_get(config, "action_dim", 4))
        hidden = int(_get(config, "hidden", 128))
        self.message_kind = message_kind
        self.message_dim = message_dim
        self.local = ResidualMLP(self.state_dim + self.action_dim + message_dim,
                                 self.state_dim, hidden)
        if message_kind == "global":
            self.global_msg = nn.Linear(self.num_objects * self.state_dim,
                                        message_dim, bias=False)
        elif message_kind == "local":
            # Slice the same initialized 4x80 matrix used by global4. Thus
            # local4 removes cross-slot inputs without changing init scale.
            weight = torch.empty(message_dim, self.num_objects * self.state_dim)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.local_msg_weight = nn.Parameter(weight.view(
                message_dim, self.num_objects, self.state_dim).permute(1, 0, 2).contiguous())

    def message(self, state):
        if self.message_kind == "global":
            msg = self.global_msg(state.flatten(1))
            return msg.unsqueeze(1).expand(-1, self.num_objects, -1)
        if self.message_kind == "local":
            return torch.einsum("bni,nki->bnk", state, self.local_msg_weight)
        return None

    def forward(self, state, actions):
        action = encode_actions(actions, self.num_objects, self.action_dim).to(state.dtype)
        features = [state, action]
        msg = self.message(state)
        if msg is not None:
            features.append(msg)
        delta = self.local(torch.cat(features, dim=-1))
        return state + delta


class FullGNN(nn.Module):
    """Official C-SWM-style fully connected directed-message transition."""
    def __init__(self, config):
        super().__init__()
        self.num_objects = int(_get(config, "num_objects", 5))
        self.state_dim = int(_get(config, "state_dim", 16))
        self.action_dim = int(_get(config, "action_dim", 4))
        hidden = int(_get(config, "hidden", 128))
        d = self.state_dim
        self.edge1 = nn.Linear(2 * d, hidden)
        self.edge2 = nn.Linear(hidden, hidden)
        self.edge_ln = nn.LayerNorm(hidden)
        self.edge3 = nn.Linear(hidden, hidden)
        pairs = [(source, target) for source in range(self.num_objects)
                 for target in range(self.num_objects) if source != target]
        self.register_buffer("edge_source", torch.tensor([p[0] for p in pairs]), persistent=False)
        self.register_buffer("edge_target", torch.tensor([p[1] for p in pairs]), persistent=False)
        self.node1 = nn.Linear(hidden + d + self.action_dim, hidden)
        self.node2 = nn.Linear(hidden, hidden)
        self.node_ln = nn.LayerNorm(hidden)
        self.node3 = nn.Linear(hidden, d)

    def forward(self, state, actions):
        n = self.num_objects
        action = encode_actions(actions, n, self.action_dim).to(state.dtype)
        source = state.index_select(1, self.edge_source)
        target = state.index_select(1, self.edge_target)
        edge_input = torch.cat((source, target), dim=-1)
        edge = F.relu(self.edge1(edge_input))
        edge = F.relu(self.edge_ln(self.edge2(edge)))
        edge = self.edge3(edge)
        aggregate = state.new_zeros(state.shape[0], n, edge.shape[-1])
        index = self.edge_source.view(1, -1, 1).expand_as(edge)
        aggregate.scatter_add_(1, index, edge)
        node = torch.cat((state, action, aggregate), dim=-1)
        delta = self.node3(F.relu(self.node_ln(self.node2(F.relu(self.node1(node))))))
        return state + delta


class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_objects = int(_get(config, "num_objects", 5))
        self.state_dim = int(_get(config, "state_dim", 16))
        self.action_dim = int(_get(config, "action_dim", 4))
        hidden = int(_get(config, "hidden", 128))
        self.token_in = nn.Linear(self.state_dim + self.action_dim, hidden)
        self.slot_embedding = nn.Parameter(torch.zeros(1, self.num_objects, hidden))
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                       dim_feedforward=512, dropout=0.0,
                                       activation="relu", batch_first=True)
            for _ in range(6)
        ])
        self.output = nn.Linear(hidden, self.state_dim)

    def forward(self, state, actions):
        action = encode_actions(actions, self.num_objects, self.action_dim).to(state.dtype)
        h = self.token_in(torch.cat((state, action), dim=-1)) + self.slot_embedding
        for layer in self.layers:
            h = layer(h)
        return state + self.output(h)


class FlatPredictor(nn.Module):
    def __init__(self, config, hidden):
        super().__init__()
        n = int(_get(config, "num_objects", 5))
        d = int(_get(config, "state_dim", 16))
        a = int(_get(config, "action_dim", 4))
        self.num_objects, self.state_dim, self.action_dim = n, d, a
        self.hidden = hidden
        self.mlp = ResidualMLP(n * (d + a), n * d, hidden)

    def forward(self, state, actions):
        action = encode_actions(actions, self.num_objects, self.action_dim).to(state.dtype)
        x = torch.cat((state, action), dim=-1).flatten(1)
        delta = self.mlp(x).view_as(state)
        return state + delta


class TransitionModel(nn.Module):
    """Adds a uniform rollout API to every predictor arm."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, actions):
        return self.model(state, actions)

    def step(self, state, actions):
        return self.forward(state, actions)

    def rollout(self, state, actions):
        """Return states after each action; actions has shape [B, horizon]."""
        states = []
        for t in range(actions.shape[1]):
            state = self.forward(state, actions[:, t])
            states.append(state)
        return torch.stack(states, dim=1)

    def rollout_final(self, state, actions):
        """Return the final state after a complete action sequence."""
        for t in range(actions.shape[1]):
            state = self.forward(state, actions[:, t])
        return state


def make_encoder(config, seed):
    with _seeded(seed):
        return SlotEncoder(config)


def _global4_budget(config):
    n = int(_get(config, "num_objects", 5))
    d = int(_get(config, "state_dim", 16))
    a = int(_get(config, "action_dim", 4))
    h = int(_get(config, "hidden", 128))
    local_input = d + a + 4
    local = (local_input + 1) * h + (h + 1) * h + (h + 1) * d
    return local + n * d * 4


def _flat_hidden(config, budget):
    n = int(_get(config, "num_objects", 5))
    d = int(_get(config, "state_dim", 16))
    a = int(_get(config, "action_dim", 4))
    inputs, outputs = n * (d + a), n * d
    return min(range(1, 4 * int(_get(config, "hidden", 128)) + 1),
               key=lambda h: (abs((inputs + 1) * h + (h + 1) * h + (h + 1) * outputs - budget), h))


def make_predictor(arm, config, seed):
    arms = {"full_gnn", "local", "local4", "global4", "global16",
            "transformer6", "flat_mlp_matched"}
    if arm not in arms:
        raise ValueError(f"unknown predictor arm: {arm}")
    with _seeded(seed):
        if arm == "full_gnn":
            model = FullGNN(config)
        elif arm == "local":
            model = LocalPredictor(config, 0, "none")
        elif arm == "local4":
            model = LocalPredictor(config, 4, "local")
        elif arm == "global4":
            model = LocalPredictor(config, 4, "global")
        elif arm == "global16":
            model = LocalPredictor(config, 16, "global")
        elif arm == "transformer6":
            model = TransformerPredictor(config)
        else:
            model = FlatPredictor(config, _flat_hidden(config, _global4_budget(config)))
        return TransitionModel(model)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_budget(config):
    """Expose the global4-matched flat width and its trainable parameter budget."""
    budget = _global4_budget(config)
    width = _flat_hidden(config, budget)
    n = int(_get(config, "num_objects", 5))
    d = int(_get(config, "state_dim", 16))
    a = int(_get(config, "action_dim", 4))
    flat = (n * (d + a) + 1) * width + (width + 1) * width + (width + 1) * n * d
    return {"global4_params": budget, "flat_hidden": width, "flat_params": flat,
            "relative_difference": (flat - budget) / budget}
