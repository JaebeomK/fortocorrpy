# Correction Methods

fortocorrpy implements six topographic-correction methods within a single
interface. All reduce reflectance variation caused by terrain-dependent
illumination, so that pixels of the same cover type compare more consistently
across slopes and aspects. Cosine, C-correction, SCS, SCS+C, and ER normalize
the illumination effect with respect to horizontal-terrain geometry, using
different assumptions about surface or canopy response. SE instead removes the
fitted illumination trend and re-centres reflectance on the mean of the
regression sample.

## The illumination condition

The local solar incidence angle `i` is the angle between the sun and the
surface normal:

```
cos i = cos(theta_s)·cos(alpha) + sin(theta_s)·sin(alpha)·cos(phi_s - beta)
```

where `theta_s` is the solar zenith, `phi_s` the solar azimuth, `alpha` the
slope, and `beta` the aspect. The horizontal illumination condition is
`cos i_h = cos(theta_s)`. SCS and SCS+C additionally include `cos(alpha)` in
their normalization to preserve sun-canopy-sensor geometry over sloping
terrain. Self-shadow corresponds to `cos i ≤ 0`.

## Regression and the C parameter

Four methods (`c`, `scsc`, `se`, `er`) require a linear regression of
reflectance against the illumination condition on the forest sample:

```
rho_t = a·cos i + b
```

`C = b / a` is a semi-empirical moderator derived from the regression. It
reduces the excessive correction the plain cosine model produces under low
illumination, and can be read as partly accounting for non-direct illumination.
Regression uses only pixels above `Config.cos_i_threshold` (default 0), so the
estimate stays physically meaningful.

The sample is not simply every illuminated pixel: it is restricted to forest
above a minimum slope, then balanced across the four aspect quadrants, and a
scene in which any quadrant holds too few eligible pixels is left uncorrected.
The criteria are in [Usage](usage.md).

## The six methods

| key      | method       | formula                                               | regression | slope | source                |
|----------|--------------|-------------------------------------------------------|------------|-------|-----------------------|
| `cosine` | Cosine       | `rho_t · cos(theta_s) / cos i`                        | no         | no    | Teillet et al. (1982) |
| `c`      | C-correction | `rho_t · (cos(theta_s) + C) / (cos i + C)`            | yes        | no    | Teillet et al. (1982) |
| `scs`    | SCS          | `rho_t · (cos(alpha)·cos(theta_s)) / cos i`           | no         | yes   | Gu & Gillespie (1998) |
| `scsc`   | SCS+C        | `rho_t · (cos(alpha)·cos(theta_s) + C) / (cos i + C)` | yes        | yes   | Soenen et al. (2005)  |
| `se`     | SE           | `rho_t − (a·cos i + b) + mean(rho)`                   | yes        | no    | Teillet et al. (1982) |
| `er`     | ER           | `rho_t − a·(cos i − cos(theta_s))`                    | yes        | no    | Tan et al. (2013)     |

### Cosine

The simplest correction: it scales reflectance by `cos(theta_s) / cos i`,
assuming reflectance is proportional to the illumination. It over-corrects as
`cos i → 0` (weakly lit slopes become excessively bright).

### C-correction

Adds the moderator `C` to the numerator and denominator of the cosine form,
which limits the over-correction at low `cos i`. `C` is estimated by regression.

### SCS

The sun-canopy-sensor correction preserves the geotropic (vertical) structure
of forest canopies by using `cos(alpha)·cos(theta_s)` in the numerator. It
needs the slope `alpha`.

### SCS+C

Combines SCS with the `C` moderator, giving the canopy-preserving geometry of
SCS and the reduced over-correction of C-correction.

### SE (Statistical-Empirical)

A subtractive method: it removes the regression trend `a·cos i + b` and
re-centres on the sample-mean reflectance `mean(rho)` over the regression
sample. The sample mean depends on the sample composition.

### ER (Empirical Rotation)

Also subtractive: it removes the regression trend but re-centres on the
horizontal illumination `cos i_h = cos(theta_s)`, whereby the intercept cancels
and the expression reduces to `rho_t − a·(cos i − cos(theta_s))`. Unlike SE, the
reference does not depend on the sample mean: it is set by the scene's own
horizontal illumination geometry.

## Application scope and self-shadow

- The regression uses only pixels with `cos i > Config.cos_i_threshold`; with
  the default threshold of 0 this excludes self-shadow and keeps the fit to
  illuminated pixels.
- Denominator forms (`cosine`, `scs`, `c`, `scsc`) are applied where
  `cos i > cos_i_threshold` (default 0); at or below it the output is NaN.
  There is no option to correct back into self-shadow: `cosine` and `scs`
  divide by `cos i`, which approaches zero at the self-shadow boundary, while
  in `c` and `scsc` the denominator `cos i + C` can approach zero. In both
  cases the correction factor diverges. A separate numerical floor rejects
  near-zero denominators, which return NaN rather than an extreme value.
- If the regression slope `a` is zero there is no illumination trend to remove.
  `C` is undefined, and the four regression methods return the input unchanged,
  with the denominator mask still applied where it belongs. `cosine` and `scs`
  do not use the coefficients and are unaffected.
- SE is the only method whose centring reference depends directly on the regression
  sample: its output is centred on the sample-mean reflectance, so changes in
  sample composition can shift that reference.
- Subtractive forms (`se`, `er`) have no denominator and are applied to all
  valid pixels regardless of the illumination threshold; results on shaded
  slopes are extrapolations of the sunlit regression and should be interpreted
  with care.

## References

- Teillet, P.M., Guindon, B., Goodenough, D.G. (1982). On the slope-aspect correction of multispectral scanner data. *Canadian Journal of Remote Sensing*, 8(2), 84–106.
- Gu, D., Gillespie, A. (1998). Topographic normalization of Landsat TM images of forest based on subpixel sun–canopy–sensor geometry. *Remote Sensing of Environment*, 64, 166–175.
- Soenen, S.A., Peddle, D.R., Coburn, C.A. (2005). SCS+C: a modified sun-canopy-sensor topographic correction in forested terrain. *IEEE Transactions on Geoscience and Remote Sensing*, 43(9), 2148–2159.
- Tan, B., et al. (2013). Improved forest change detection with terrain illumination corrected Landsat images. *Remote Sensing of Environment*, 136, 469–483.