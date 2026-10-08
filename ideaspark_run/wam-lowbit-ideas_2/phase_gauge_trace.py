"""A small algebra check, not a model or task experiment."""
from fractions import Fraction

draws = [Fraction(i, 4) for i in range(4)]
phase = [Fraction(0), Fraction(1, 2)]
permuted = list(reversed(phase))

def joint_noise(table):
    return sorted(tuple((u + p) % 1 for p in table) for u in draws)

print("original joint dithers:", joint_noise(phase))
print("permuted joint dithers:", joint_noise(permuted))
print("identical joint law:", joint_noise(phase) == joint_noise(permuted))
