# m35_photo_edit

Restoring badly fogged 35mm colour-negative scans from a Kodak M35. Classical image
processing only — no generative models, no invented detail.

## install

```
pip install -r requirements.txt
```

## run

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

`--help` lists the rest.

## layout

```
m35/analysis.py       measure the frame, pick a correction class
m35/color_cast.py     three-point per-channel match, fog flattening
m35/tone.py           tone curve, CLAHE
m35/color_grade.py    saturation shaping, warm grade
m35/noise.py          chroma denoise, confidence gate, sharpening
m35/render.py         the neutral / film / bw variants
m35/pipeline.py       per-file driver
```

## note

Every frame on this roll grades `recovery: poor` — red and green survive in only 37–102
code values out of 255. The `_bw` outputs are the better rescue. Diagnostics are printed
per frame.
