from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS

FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "Arial.ttf",
    "Helvetica.ttc",
    "LiberationSans-Regular.ttf",
)


def load_font(size: int):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_contact_sheet(rows: list[dict], out_path: Path, thumb_w: int) -> None:
    if not rows:
        return

    cols = ["original", "neutral", "film", "bw"]
    titles = ["ORIGINAL", "NEUTRAL", "FILM", "BLACK & WHITE"]
    pad, header_h, label_h, caption_h = 14, 54, 26, 34

    thumbs: list[list[Image.Image]] = []
    for r in rows:
        row_imgs = []
        for c in cols:
            im = Image.fromarray(r[c], mode="RGB")
            h = max(1, int(round(im.height * thumb_w / im.width)))
            row_imgs.append(im.resize((thumb_w, h), RESAMPLE_LANCZOS))
        thumbs.append(row_imgs)

    row_h = [max(i.height for i in row) for row in thumbs]
    sheet_w = pad + len(cols) * (thumb_w + pad)
    sheet_h = header_h + label_h + sum(h + caption_h + pad for h in row_h) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (22, 22, 24))
    draw = ImageDraw.Draw(sheet)
    f_title = load_font(24)
    f_head = load_font(17)
    f_cap = load_font(14)

    draw.text((pad, 16), "35mm film restoration -- contact sheet",
              font=f_title, fill=(238, 238, 240))
    for ci, t in enumerate(titles):
        draw.text((pad + ci * (thumb_w + pad), header_h + 4), t, font=f_head, fill=(150, 200, 235))

    y = header_h + label_h
    for row, rh, meta in zip(thumbs, row_h, rows):
        for ci, im in enumerate(row):
            x = pad + ci * (thumb_w + pad)
            sheet.paste(im, (x, y + (rh - im.height) // 2))
        cap = (f"{meta['name']}   |   class: {meta['preset']}   |   cast: {meta['cast']} "
               f"({meta['cast_ratio']:.2f}x)   |   worst channel range: "
               f"{meta['min_range']:.0f}/255   |   recovery: {meta['recoverability']}")
        draw.text((pad, y + rh + 8), cap, font=f_cap, fill=(168, 168, 174))
        y += rh + caption_h + pad

    sheet.save(out_path, format="JPEG", quality=92, optimize=True)
