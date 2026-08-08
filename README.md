# Kodak M35 Analog Camera -- Photo Editing 

Restoring badly fogged 35mm colour-negative scans from a Kodak M35. Classical image
processing only — no generative models, no invented detail.

## install

```
pip install -r requirements.txt
```

## restore

```
python restore_film.py ./scans ./restored
```

Writes `<name>_neutral.jpg`, `<name>_film.jpg`, `<name>_bw.jpg` and a `contact_sheet.jpg`.
Originals are read-only; the output folder must differ from the input folder.

```
--force-preset {light,moderate,severe,extreme}
--report diagnostics.json
--recursive --jobs N --quality N
```

## experiment

```
python experiment.py ./scans ./experiments
```

Seven looks per frame, written alongside `restored/` and never over it.

`fused` is the interesting one. The B&W conversion is much cleaner than the colour
luminance, because it is built from an information-weighted channel mix instead of a
standard luma mix. So `fused` takes luminance from there, chroma from the colour pipeline,
and multiplies the chroma back up — the chroma gain is weighted by the same confidence map
used in the restore, so dead fog regions do not get their residual cast amplified along
with the real colour. Multi-scale Laplacian local contrast on the luminance, then halation,
grain and a slight vignette.

`fused_haze` adds dark-channel-prior dehazing. `bleach` and `cross` are colour grades.
`selenium`, `sepia` and `cyanotype` are split-toned from the fused luminance — on this roll
they are the best-looking outputs by some margin.

### gradient-domain seam removal

A light leak often lands as a hard, frame-spanning step. Chroma flattening cannot follow it
and a Gaussian estimate smears across it, so the band survives every filter-based attempt.

`m35/lab/recon.py` finds those steps by their robust z-score in the column/row gradient
profile, zeroes them in the gradient field, and reintegrates the image by solving Poisson
with Neumann boundaries via DCT. The step vanishes and the two sides merge seamlessly;
nothing else in the frame moves.

The threshold matters. Genuine leak seams on this roll score z = 196, 118 and 51; real
image edges score 12–20. At `--seam-z 12` the solver flattens actual content. The default
of 40 fires on two frames out of eight, which is the correct answer.

Dust spotting exists (`--spot-z`) but is **off by default**. At this resolution grain and
dust are not separable — tightening the threshold from 5 to 20 takes the region count from
11084 to 477 with no plateau, which is a continuous grain distribution rather than a
discrete set of defects. Enabling it removes grain, not dust.

```
--chroma-gain N --lc-gain N --grain N --halation N --dehaze-strength N
--seam-z N --spot-z N
```

## layout

```
m35/analysis.py         measure the frame, pick a correction class
m35/color_cast.py       three-point per-channel match, fog flattening
m35/tone.py             tone curve, CLAHE
m35/color_grade.py      saturation shaping, warm grade
m35/noise.py            chroma denoise, confidence gate, sharpening
m35/render.py           the neutral / film / bw variants
m35/pipeline.py         per-file driver
m35/lab/fusion.py       luminance/chroma fusion
m35/lab/local_contrast.py   Laplacian local contrast, dehazing
m35/lab/film_look.py    halation, grain, vignette
m35/lab/grades.py       toning and grading looks
m35/lab/recon.py        Poisson solver, seam removal, defect spotting
```

## note

Every frame on this roll grades `recovery: poor` — red and green survive in only 37–102
code values out of 255. Diagnostics are printed per frame. `restore_film.py` is the honest
restoration; `experiment.py` is where the same data gets pushed for looks.
