"""Small checks that distinguish coherent bias from cancelling action errors."""
import torch
from aggregate import command_errors, means, episode_means, observed_features


def main():
    zero = torch.zeros(32, 7)
    alternating = zero.clone()
    alternating[:10:2, 0] = .1
    alternating[1:10:2, 0] = -.1
    coherent = zero.clone()
    coherent[:10, 0] = .08
    a, b = command_errors(torch, alternating, zero), command_errors(torch, coherent, zero)
    assert b['command_rmse'] < a['command_rmse']
    assert b['command_prefix_cumsum_max_abs'] > a['command_prefix_cumsum_max_abs']
    assert b['command_chunk_bias_rms'] > a['command_chunk_bias_rms']
    assert all(value == 0 for value in command_errors(torch, zero, zero).values())
    result = means([{'suite': 'a', 'x': 1}, *[{'suite': 'b', 'x': 3} for _ in range(3)]], ['x'])
    assert result['equal_suite_mean']['x'] == 2
    result = episode_means([{'index': 1, 'x': 1}, *[{'index': 2, 'x': 3} for _ in range(3)]], ['x'])
    assert result['episodes'] == 2 and result['states'] == 4
    assert result['episode_mean']['x'] == 2
    future = torch.zeros(1, 48, 3, 14, 28)
    future[:, :, 0, :, 14:] = 1
    feature = observed_features(torch, future)
    assert torch.equal(feature[:96], torch.zeros(96))
    assert torch.equal(feature[96:144], torch.ones(48))
    assert torch.equal(feature[144:], torch.zeros(48))
    future[:, :, 1:] = 1000
    assert torch.equal(feature, observed_features(torch, future))
    print('Aggregate checks passed: command bias, episode weights, camera separation and observed-only features')


if __name__ == '__main__':
    main()
