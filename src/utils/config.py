"""Experiment configurations."""

from types import SimpleNamespace

# Global constants
C = SimpleNamespace(
    verbose=True,
    n_bins=20,
    n_runs=30,
    n_shadow_runs=50,
    shadow_train_size=10_000,
    shadow_epsilons=[0.1, 0.32, 1.0, 3.16, 10.0, 31.62, 100.0, 316.23, 1000.0],
)

# Focal-point weight thresholds (epsilon -> threshold fraction)
FP_WEIGHT_THRESHOLDS = {
    0.0: 0.4,
    0.32: 0.5,
    1.0: 0.6,
    3.16: 0.65,
    10.0: 0.7,
    31.62: 0.75,
    100.0: 0.8,
    316.23: 0.85,
    1000.0: 0.85,
}


def get_weight_threshold(eps):
    """Get the FP weight threshold for a given epsilon."""
    threshold = 0.4
    for e, t in sorted(FP_WEIGHT_THRESHOLDS.items()):
        if eps >= e:
            threshold = t
    return threshold


class ExperimentConfig:
    """Configuration for a MAMA-MIA experiment."""

    def __init__(
        self,
        data_name="snake",
        train_size=1000,
        synth_size=None,
        num_targets=32,
        epsilons=None,
        overlapping_aux=True,
        set_MI=False,
        n_shadow_runs=50,
        partition="centralized",  # "centralized", "horizontal", "vertical", "mixed"
        n_holders=2,
    ):
        self.data_name = data_name
        self.train_size = train_size
        self.synth_size = synth_size or train_size
        self.num_targets = num_targets
        self.epsilons = epsilons or C.shadow_epsilons
        self.overlapping_aux = overlapping_aux
        self.set_MI = set_MI
        self.n_shadow_runs = n_shadow_runs
        self.partition = partition
        self.n_holders = n_holders

    def __repr__(self):
        return (
            f"ExperimentConfig(data={self.data_name}, n={self.train_size}, "
            f"partition={self.partition}, holders={self.n_holders})"
        )
