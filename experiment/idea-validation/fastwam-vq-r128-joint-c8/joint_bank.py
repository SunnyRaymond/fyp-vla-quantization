"""Train rotations and encoded VQ parameters through the deployed student."""
import torch
import torch.nn.functional as F

from c8_train_banks import DecodedWeight, TrainableBanks
from rotation_bank import A4, Rotations, stream_for, transformed


class JointBanks(TrainableBanks):
    def __init__(self, runtime, directory):
        super().__init__(runtime, directory)
        self.rotations = Rotations(self.modules)
        for name, mod in self.modules.items():
            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                qx = A4.apply(transformed(x, self.transforms[_name], self.rotations, stream_for(_name)))
                weight = DecodedWeight.apply(self.books[_name], self.logs[_name], self.active[_name],
                    self.encodings[_name]['indices'], self.encodings[_name]['scale_exponents'], _mod.weight)
                return F.linear(qx, weight, _mod.bias)
            mod.forward = forward

    def vq_parameters(self):
        return super().parameters()

    def parameters(self):
        return self.vq_parameters() + list(self.rotations.parameters())

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.forwards[name]
