"""Categorical-palette validator (dataviz skill's job, computed locally
because the bundled validator wasn't present). For every pair of colors
it reports OKLab dE x100 under normal vision and under Machado-2009
CVD simulation (protan/deutan/tritan, severity 1.0). Rules:
  normal-vision floor: min pair dE >= 15  (full-color readers)
  CVD separation:      target >= 8, hard floor 6 (else secondary encoding)
"""
import sys
from itertools import combinations

PAL = sys.argv[1:] or [  # shipped trader palette (Okabe-Ito, validated)
    "#d55e00", "#0072b2", "#009e73", "#e69f00",
    "#56b4e9", "#cc79a7", "#f0e442", "#000000"]

MACHADO = {  # severity 1.0, applied in linear RGB
 "protan": [[0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998]],
 "deutan": [[0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881]],
 "tritan": [[1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900]]}


def hex2lin(h):
    h = h.lstrip("#")
    srgb = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return [(c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4) for c in srgb]


def mul(m, v):
    return [sum(m[i][j]*v[j] for j in range(3)) for i in range(3)]


def lin2oklab(r, g, b):
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l_, m_, s_ = l**(1/3), m**(1/3), s**(1/3)
    return (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
            1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
            0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_)


def de(a, b):
    return 100*sum((x-y)**2 for x, y in zip(a, b))**0.5


def oklab_of(h, cvd=None):
    lin = hex2lin(h)
    if cvd:
        lin = [max(0.0, min(1.0, x)) for x in mul(MACHADO[cvd], lin)]
    return lin2oklab(*lin)


def main():
    modes = [None, "protan", "deutan", "tritan"]
    labels = {None: "normal", "protan": "protan", "deutan": "deutan",
              "tritan": "tritan"}
    lab = {m: {h: oklab_of(h, m) for h in PAL} for m in modes}
    print(f"palette ({len(PAL)}): {', '.join(PAL)}\n")
    worst = {}
    for m in modes:
        pairs = [(a, b, de(lab[m][a], lab[m][b]))
                 for a, b in combinations(PAL, 2)]
        lo = min(pairs, key=lambda x: x[2])
        worst[m] = lo
        floor = 15 if m is None else 8
        tag = "PASS" if lo[2] >= floor else ("WARN" if lo[2] >= 6 else "FAIL")
        print(f"{labels[m]:7s} min pair dE = {lo[2]:5.1f}  "
              f"({lo[0]}~{lo[1]})  need>={floor}  [{tag}]")
    nv = worst[None][2]
    cvd_min = min(worst[m][2] for m in modes if m)
    verdict = ("PASS" if nv >= 15 and cvd_min >= 8 else
               "USE_SECONDARY_ENCODING" if nv >= 15 and cvd_min >= 6 else
               "FAIL")
    print(f"\nVERDICT: {verdict}  (normal {nv:.1f}, worst-CVD {cvd_min:.1f})")
    if verdict != "PASS":
        print("note: line charts already carry a secondary channel "
              "(direct end-labels + legend), which the skill allows for "
              "CVD dE in [6,8).")


if __name__ == "__main__":
    main()
