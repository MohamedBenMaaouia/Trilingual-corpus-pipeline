"""The LSH S-curve of a 128-value signature for several band splits (plan 4.5.1, S4-07).

    python -m corpus.review.lsh_curve [--out docs/figures/sprint-4-lsh-curve.png]

Plots P(candidate) = 1 - (1 - s^rows)^bands against the Jaccard similarity s (spec 3d)
for 32 x 4, 16 x 8, 8 x 16 and 4 x 32. The two settings compared on the dev run are in
colour, with a dot where each one's threshold t = (1/bands)^(1/rows) sits; the other
two are context, in gray. Needs matplotlib, a dev dependency: run it in the dev
environment, never in a job.
"""

import argparse

from corpus.dedup.banding import LshParams, candidate_probability

SPLITS = [(32, 4), (16, 8), (8, 16), (4, 32)]  # threshold order, left to right
COMPARED = {(16, 8): "#2a78d6", (8, 16): "#eb6834"}  # categorical slots 1-2, validated
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
DPI = 144
LINE = 2 * 72 / DPI  # a 2-pixel line, in points
# Where each curve's name sits: (height on the curve, side). 4 x 32 is named on its
# left, lower down: on its right it would leave the plot.
NAME_AT = {
    (32, 4): (0.5, "right"),
    (16, 8): (0.5, "right"),
    (8, 16): (0.5, "right"),
    (4, 32): (0.3, "left"),
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # files only, no window
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/figures/sprint-4-lsh-curve.png")
    args = parser.parse_args()

    xs = [i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for bands, rows in SPLITS:
        colour = COMPARED.get((bands, rows), AXIS)
        chosen = " (chosen)" if (bands, rows) == (LshParams.bands, LshParams.rows) else ""
        ys = [candidate_probability(x, bands, rows) for x in xs]
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=LINE,
            label=f"{bands} x {rows}{chosen}",
            zorder=3 if colour != AXIS else 2,
        )
        # The curve's name, beside it where it crosses NAME_AT's height.
        height, side = NAME_AT[(bands, rows)]
        x_at = next(x for x, y in zip(xs, ys, strict=True) if y >= height)
        ax.annotate(
            f"{bands} x {rows}",
            (x_at, height),
            xytext=(6 if side == "right" else -6, -3),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            color=MUTED,
            fontsize=8,
        )
        if colour == AXIS:
            continue
        # The compared settings: a dot at the threshold, labelled on its left, where
        # the space between the curves is free.
        t = LshParams(bands, rows).threshold
        at_t = candidate_probability(t, bands, rows)
        ax.plot(
            [t],
            [at_t],
            "o",
            color=colour,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
        ax.annotate(
            f"t = {t:.2f}\nP = {at_t:.2f}",
            (t, at_t),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Jaccard similarity of two pages (word 5-grams)", color=MUTED)
    ax.set_ylabel("P(the pair becomes a candidate)", color=MUTED)
    ax.set_title(
        "LSH banding of a 128-value MinHash signature: bands x rows",
        color=INK,
        loc="left",
        fontsize=11,
    )
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=INK)
    fig.text(
        0.01,
        0.01,
        "P = 1 - (1 - s^rows)^bands. Candidates are then kept only if their signatures agree on "
        "at least t of their values.",
        color=MUTED,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"lsh_curve: wrote {args.out}")


if __name__ == "__main__":
    main()
