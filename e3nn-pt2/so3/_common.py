"""Common utility functions used in the so3 submodule."""

def _check_degree_is_positive_or_zero(degree: int) -> None:
  """Checks whether the input degree is positive or zero."""
  if degree < 0:
    raise ValueError(f'degree must be positive or zero, received {degree}')

