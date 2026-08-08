# Bibliography — snow-mcrt

Canonical references for light propagation in snow and Monte Carlo radiative transfer.

> Flags: **(foundational)** textbook or pre-internet classic · **(benchmark)** used for validation · **(code)** open-source reference implementation.

## Foundational physics

- **(foundational)** Bohren, C. F., & Huffman, D. R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley. ISBN 978-0-471-05772-3.
- **(foundational)** van de Hulst, H. C. (1981). *Light Scattering by Small Particles*. Dover. ISBN 978-0-486-64228-4.
- **(foundational)** Chandrasekhar, S. (1960). *Radiative Transfer*. Dover.
- Henyey, L. G., & Greenstein, J. L. (1941). Diffuse radiation in the galaxy. *Astrophysical Journal*, 93, 70–83. DOI: [10.1086/144246](https://doi.org/10.1086/144246)

## Snow optics — canon

- **(benchmark)** Warren, S. G. (1982). Optical properties of snow. *Reviews of Geophysics*, 20(1), 67–89. DOI: [10.1029/RG020i001p00067](https://doi.org/10.1029/RG020i001p00067)
- **(benchmark)** Wiscombe, W. J., & Warren, S. G. (1980). A model for the spectral albedo of snow. I: Pure snow. *Journal of the Atmospheric Sciences*, 37(12), 2712–2733. DOI: [10.1175/1520-0469(1980)037\<2712:AMFTSA\>2.0.CO;2](https://doi.org/10.1175/1520-0469(1980)037%3C2712:AMFTSA%3E2.0.CO;2)
- **(benchmark)** Warren, S. G., & Wiscombe, W. J. (1980). A model for the spectral albedo of snow. II: Snow containing atmospheric aerosols. *Journal of the Atmospheric Sciences*, 37(12), 2734–2745. DOI: [10.1175/1520-0469(1980)037\<2734:AMFTSA\>2.0.CO;2](https://doi.org/10.1175/1520-0469(1980)037%3C2734:AMFTSA%3E2.0.CO;2)
- Warren, S. G., & Brandt, R. E. (2008). Optical constants of ice from the ultraviolet to the microwave: A revised compilation. *Journal of Geophysical Research: Atmospheres*, 113, D14220. DOI: [10.1029/2007JD009744](https://doi.org/10.1029/2007JD009744)
- Warren, S. G. (2019). Optical properties of ice and snow. *Philosophical Transactions of the Royal Society A*, 377, 20180161. DOI: [10.1098/rsta.2018.0161](https://doi.org/10.1098/rsta.2018.0161)
- Kokhanovsky, A. A., & Zege, E. P. (2004). Scattering optics of snow. *Applied Optics*, 43(7), 1589–1602. DOI: [10.1364/AO.43.001589](https://doi.org/10.1364/AO.43.001589)

## Remote sensing — bands, indices and retrieval

Supporting [`docs/remote-sensing.md`](../remote-sensing.md).

- **(benchmark)** Dozier, J. (1989). Spectral signature of alpine snow cover from the Landsat Thematic Mapper. *Remote Sensing of Environment*, 28, 9–22. DOI: [10.1016/0034-4257(89)90101-6](https://doi.org/10.1016/0034-4257\(89\)90101-6) — where the visible/SWIR contrast becomes an index.
- **(benchmark)** Hall, D. K., Riggs, G. A., & Salomonson, V. V. (1995). Development of methods for mapping global snow cover using Moderate Resolution Imaging Spectroradiometer data. *Remote Sensing of Environment*, 54(2), 127–140. DOI: [10.1016/0034-4257(95)00137-P](https://doi.org/10.1016/0034-4257\(95\)00137-P) — the operational NDSI > 0.4 threshold.
- Nolin, A. W., & Dozier, J. (2000). A hyperspectral method for remotely sensing the grain size of snow. *Remote Sensing of Environment*, 74(2), 207–216. DOI: [10.1016/S0034-4257(00)00111-5](https://doi.org/10.1016/S0034-4257\(00\)00111-5)
- Painter, T. H., et al. (2009). Retrieval of subpixel snow-covered area, grain size, and albedo from MODIS. *Remote Sensing of Environment*, 113(4), 868–879. DOI: [10.1016/j.rse.2009.01.001](https://doi.org/10.1016/j.rse.2009.01.001)
- Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25–36. DOI: [10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026) — the band definitions in `domain/sensor.py`.

## Open reference implementations

- **(code)** **SNICAR** — Flanner, M. G., & Zender, C. S. (2005, 2006). Snowpack radiative transfer model. Web: https://snow.engin.umich.edu
- **(code, benchmark)** Flanner, M. G., et al. (2007). Present-day climate forcing and response from black carbon in snow. *Journal of Geophysical Research*, 112, D11202. DOI: [10.1029/2006JD008003](https://doi.org/10.1029/2006JD008003)
- **(code, benchmark)** **TARTES** — Libois, Q., Picard, G., et al. (2013). Influence of grain shape on light penetration in snow. *The Cryosphere*, 7, 1803–1818. DOI: [10.5194/tc-7-1803-2013](https://doi.org/10.5194/tc-7-1803-2013) · Code: https://gp.snow.univ-grenoble-alpes.fr/tartes/

## Monte Carlo methodology (general photon transport)

- **(code, foundational)** Wang, L., Jacques, S. L., & Zheng, L. (1995). MCML — Monte Carlo modeling of light transport in multi-layered tissues. *Computer Methods and Programs in Biomedicine*, 47(2), 131–146. DOI: [10.1016/0169-2607(95)01640-F](https://doi.org/10.1016/0169-2607(95)01640-F)
- Prahl, S. A., Keijzer, M., Jacques, S. L., & Welch, A. J. (1989). A Monte Carlo model of light propagation in tissue. *SPIE Proceedings IS*, 5, 102–111.

## Impurities in snow

- Dang, C., Brandt, R. E., & Warren, S. G. (2015). Parameterizations for narrowband and broadband albedo of pure snow and snow containing mineral dust and black carbon. *Journal of Geophysical Research: Atmospheres*, 120, 5446–5468. DOI: [10.1002/2014JD022646](https://doi.org/10.1002/2014JD022646)
- Skiles, S. M., et al. (2018). Radiative forcing by light-absorbing particles in snow. *Nature Climate Change*, 8, 964–971. DOI: [10.1038/s41558-018-0296-5](https://doi.org/10.1038/s41558-018-0296-5)

## Optical constants datasets

- Warren & Brandt (2008) — tabulated ice refractive index (see above).
- refractiveindex.info — ice entry: https://refractiveindex.info/?shelf=3d&book=crystals&page=ice

## arXiv-track — preprints to curate

Search strategy, run each periodically and pick what lands:

- arXiv `physics.ao-ph`: `snow albedo monte carlo`, `snow radiative transfer`, `BRDF snow`
- arXiv `physics.comp-ph`: `photon transport snow`, `Mie scattering GPU`, `vectorized Monte Carlo`
- arXiv `physics.optics`: `multiple scattering particulate media`, `radiative transfer granular`

Non-arXiv but open-access journals worth monitoring: *The Cryosphere* (Copernicus), *Remote Sensing of Environment*, *Applied Optics*, *Journal of Quantitative Spectroscopy and Radiative Transfer*.

> TODO: populate with 5–10 curated recent references after the first reading pass.

---

**Last updated:** 2026-04-19
