"""GPU entry point reserved for a future GPU-native RBF implementation."""

import warnings

from train_evaluate import main


if __name__ == "__main__":
    warnings.warn(
        "The current RBFNetwork uses NumPy, SciPy, and scikit-learn; training runs on CPU. "
        "This entry point is retained so a GPU-native implementation can be introduced without changing commands.",
        RuntimeWarning,
        stacklevel=1,
    )
    main()
