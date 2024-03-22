from tinygrad import Tensor
import BasisLib
import tinye3nn

class SphericalHarmonics:
  def __init__(
      self,
      max_degree: int,
      cartesian_order: bool = False,
      normalization: str = "4pi"
    ):
  
      # Perform checks
      BasisLib.so3.common._check_degree_is_positive_or_zero(max_degree)
    
      # Load/Generate lookup table and convert to jax array.
      lookup_table = BasisLib.so3.generate_spherical_harmonics_lookup_table(max_degree)
      cm = lookup_table['cm']
      ls = lookup_table['ls']
      # Apply normalization constants.
      for l in range(max_degree + 1):
        cm[:, l**2 : (l + 1) ** 2] *= BasisLib.so3.normalization_constant(normalization, l)
    
      # Optionally reorder spherical harmonics to Cartesian order.
      if cartesian_order:
        cm = cm[:, BasisLib.ops.cartesian_permutation(max_degree)]
      
      self.max_degree = max_degree
      self.cm = Tensor(cm, requires_grad=False)
      self.ls = Tensor(ls, requires_grad=False)

  def __call__(
      self,
      r,
      r_is_normalized: bool = True):

    # Perform checks.
    if r.shape[-1] != 3:
      raise ValueError(f'r must have shape (..., 3), received shape {r.shape}')

    # Normalize r (if not already normalized).
    if not r_is_normalized:
      r = tinye3nn.ops.normalize(r, axis=-1)

    # Calculate all relevant monomials in the (x, y, z)-coordinates.
    # Note: This is done via integer powers and indexing on purpose! Using
    # jnp.power or the "**"-operator for this operation leads to NaNs in the
    # gradients for some inputs (jnp.power is not NaN-safe)
    r_powers = tinye3nn.so3.integer_powers(r.unsqueeze(-2), self.max_degree)
    monomials = (
          r_powers[..., 0][..., self.ls[:, 0]]  #   x**lx.
          * r_powers[..., 1][..., self.ls[:, 1]]  # y**ly.
          * r_powers[..., 2][..., self.ls[:, 2]]  # z**lz.
      )

    # Calculate and return spherical harmonics (linear combination of monomials).
    return Tensor.matmul(monomials, self.cm)