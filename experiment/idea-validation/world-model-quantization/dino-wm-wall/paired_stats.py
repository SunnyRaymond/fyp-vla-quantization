"""Small-sample exact binomial summaries; no independent-pairs assumption."""
from math import comb


def binomial_cdf(k, n, probability):
    return sum(comb(n, i) * probability**i * (1 - probability)**(n-i)
               for i in range(k + 1))


def binomial_interval(k, n, alpha=0.05):
    """Two-sided Clopper-Pearson interval by inversion of binomial tails."""
    assert 0 <= k <= n and n > 0
    def solve(cutoff, target):
        low, high = 0.0, 1.0
        for _ in range(70):
            middle = (low + high) / 2
            if binomial_cdf(cutoff, n, middle) > target:
                low = middle
            else:
                high = middle
        return (low + high) / 2
    lower = 0.0 if k == 0 else solve(k - 1, 1 - alpha/2)
    upper = 1.0 if k == n else solve(k, alpha/2)
    return [lower, upper]


def paired_summary(wins, losses, n):
    # Each category is a binomial marginal. Bonferroni joint coverage of their
    # 97.5% intervals gives a conservative 95% interval for P(win)-P(loss).
    win_interval = binomial_interval(wins, n, alpha=0.025)
    loss_interval = binomial_interval(losses, n, alpha=0.025)
    discordant = wins + losses
    pvalue = min(1.0, 2 * binomial_cdf(min(wins, losses), discordant, 0.5)) if discordant else 1.0
    return {'wins': wins, 'losses': losses, 'difference': (wins-losses)/n,
            'difference_ci95_conservative': [win_interval[0]-loss_interval[1],
                                              win_interval[1]-loss_interval[0]],
            'ci_method': 'Bonferroni joint Clopper-Pearson intervals for paired win/loss probabilities',
            'mcnemar_exact_two_sided_p': pvalue,
            'multiplicity': 'individual comparison; no familywise significance claim'}
