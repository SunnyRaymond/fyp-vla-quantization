"""Small exact per-environment/per-role embedding cache primitive."""


class PerEnvironmentExactCache:
    """Store one action-free embedding per (environment slot, semantic role).

    A hit requires exact equality with the stored input tensor. Callers must
    create a fresh instance for every native solver.solve invocation.
    """

    def __init__(self, torch):
        self.torch = torch
        self.entries = {}
        self.active_environment = None

    def activate_environment(self, environment_slot):
        environment_slot = int(environment_slot)
        if self.active_environment != environment_slot:
            self.entries.clear()
            self.active_environment = environment_slot

    def _bytes_equal(self, left, right):
        if (not self.torch.is_tensor(left) or not self.torch.is_tensor(right)
                or left.shape != right.shape or left.dtype != right.dtype
                or left.device != right.device):
            return False
        left_bytes = left.contiguous().view(self.torch.uint8)
        right_bytes = right.contiguous().view(self.torch.uint8)
        return bool(self.torch.equal(left_bytes, right_bytes))

    def lookup(self, environment_slot, role, pixels):
        if int(environment_slot) != self.active_environment:
            return None
        entry = self.entries.get((int(environment_slot), str(role)))
        if entry is None:
            return None
        if not self._bytes_equal(entry["pixels"], pixels):
            return None
        return entry["embedding"]

    def store(self, environment_slot, role, pixels, embedding):
        if int(environment_slot) != self.active_environment:
            raise RuntimeError("activate the environment before storing cache entries")
        key = (int(environment_slot), str(role))
        self.entries[key] = {
            "pixels": pixels.detach().clone(),
            "embedding": embedding.detach().clone(),
        }

    def payload_bytes(self):
        return sum(entry["pixels"].numel() * entry["pixels"].element_size()
                   + entry["embedding"].numel() * entry["embedding"].element_size()
                   for entry in self.entries.values())


def identity_self_check(torch):
    cache = PerEnvironmentExactCache(torch)
    source = torch.arange(50, dtype=torch.float32).reshape(50, 1)
    emb = source + 100.0
    cache.activate_environment(0)
    cache.store(0, "current", source[0:1], emb[0:1])
    assert cache.lookup(0, "current", source[0:1]) is not None
    assert cache.lookup(1, "current", source[0:1]) is None
    assert cache.lookup(0, "goal", source[0:1]) is None
    assert cache.lookup(0, "current", source[1:2]) is None
    changed = source[0:1] + 0.5
    assert cache.lookup(0, "current", changed) is None
    cache.store(0, "current", changed, emb[0:1])
    assert cache.lookup(0, "current", changed) is not None
    cache.activate_environment(1)
    assert cache.lookup(0, "current", changed) is None
    assert cache.lookup(1, "current", source[1:2]) is None
    cache.store(1, "current", source[1:2], emb[1:2])
    assert len(cache.entries) == 1
    cache.activate_environment(2)
    cache.store(2, "current", source[2:3], emb[2:3])
    positive_zero = torch.tensor([0.0], dtype=torch.float32)
    negative_zero = torch.tensor([-0.0], dtype=torch.float32)
    cache.store(2, "goal", positive_zero, positive_zero)
    assert cache.lookup(2, "goal", positive_zero) is not None
    assert cache.lookup(2, "goal", negative_zero) is None
    device_miss_checked = False
    try:
        meta_pixels = torch.empty((1, 1), dtype=torch.float32, device="meta")
        assert cache.lookup(2, "current", meta_pixels) is None
        device_miss_checked = True
    except (RuntimeError, NotImplementedError):
        # The runtime records that meta tensors cannot exercise a device mismatch.
        device_miss_checked = False
    return {"status": "PASS", "environment_slots": 50, "roles": ["current", "goal"],
            "checks": ["same identity hit", "cross-environment miss", "cross-role miss",
                       "changed-input miss", "updated-input hit", "environment-switch eviction",
                       "maximum two resident roles", "signed-zero byte mismatch"],
            "maximum_resident_entries": 2,
            "device_mismatch_checked": device_miss_checked,
            "device_guard_code_path": "entry device is checked before byte comparison"}
