# Source and version inventory

> **Acquisition / verification date:** 2026-08-31  
> **Total:** 12 PDFs, 198,566,678 bytes (189.37 MiB)  
> **Canonical reading choice:** venue final when directly accessible; otherwise an explicitly versioned arXiv file

## Validation rule

Every local PDF below passed the following checks:

1. first five bytes are `%PDF-`;
2. `pypdf` reopened the file and confirmed it is unencrypted;
3. page count was recorded;
4. extracted first-page text matched the expected title;
5. representative first and method pages were rendered and visually inspected for readability;
6. SHA-256 was recorded.

This validates acquisition and structure. It does not mean every experimental claim has been independently replicated.

## Local files

| Role | Work | Local file | Source/version boundary | Pages | Bytes | SHA-256 |
|---|---|---|---|---:|---:|---|
| Core | REPA | [repa-iclr-2025-final.pdf](repa-iclr-2025-final.pdf) | ICLR 2025 official proceedings final; canonical local version | 43 | 34,693,389 | `d97efd44d9354162bc4ad9b0d5288952f09a90e99c5da344b2098204b7a8164a` |
| Core | iREPA | [irepa-iclr-2026-final.pdf](irepa-iclr-2026-final.pdf) | ICLR 2026 official proceedings final; canonical local version | 39 | 38,920,911 | `962345b71fae7157c39708fcf386d3fffca8ab6ebac6b1686aeb6db622adb967` |
| Direct prior art | FLARE | [prior-art-flare-corl-2025-final.pdf](prior-art-flare-corl-2025-final.pdf) | PMLR / CoRL 2025 final | 20 | 10,467,300 | `fc769418d2e1de506340ec3ac7d289c74a2fb6462a815bcc894d403a55851c6e` |
| Direct prior art | Spatial Forcing | [prior-art-spatial-forcing-arxiv-v2.pdf](prior-art-spatial-forcing-arxiv-v2.pdf) | arXiv:2510.12276 v2; official venue asset returned 403 during acquisition | 19 | 1,990,671 | `c7d15ca562f5639ab88c60e7de0de4c34965e3c57710e79febfa249b8bddada5` |
| Direct prior art | FRAPPE | [prior-art-frappe-arxiv-v1.pdf](prior-art-frappe-arxiv-v1.pdf) | arXiv:2602.17259 v1 | 16 | 27,324,107 | `12eac2e5cd9812e23f337f94ddd8081f8805532d9904658d366952cabc46faa1` |
| Partial/direct prior art | FutureVLA | [prior-art-futurevla-arxiv-v1.pdf](prior-art-futurevla-arxiv-v1.pdf) | arXiv:2603.10712 v1 | 30 | 15,622,212 | `46b3cf66ea3f63be29e7afb02e0175682f9ccc07c5622f6e19da8f5c50da1017` |
| Direct prior art | VEGA | [prior-art-vega-arxiv-v1.pdf](prior-art-vega-arxiv-v1.pdf) | arXiv:2605.10485 v1 | 18 | 7,839,224 | `635569d092e7e7ccdfca9b25cd70411dadee6b8edd812ca52e9c62fe674d0e6b` |
| Direct WAM prior art | AGRA | [prior-art-agra-arxiv-v1.pdf](prior-art-agra-arxiv-v1.pdf) | arXiv:2606.12217 v1 | 23 | 18,149,982 | `8fe740c3e5496a1174636e496410b954e5b63928181f5906f40c959123c72389` |
| Direct prior art | SAM3D-Guided | [prior-art-sam3d-vla-arxiv-v1.pdf](prior-art-sam3d-vla-arxiv-v1.pdf) | arXiv:2607.25912 v1 | 11 | 5,554,248 | `f7a131d01582af371b1dc98ce8c5980ba7870136f3f59f91b6dcafad9b169e28` |
| Direct WAM prior art | Robust-WAM | [prior-art-robust-wam-arxiv-v1.pdf](prior-art-robust-wam-arxiv-v1.pdf) | arXiv:2608.05903 v1 | 13 | 2,960,753 | `6d5ad91cc2d4785550e525bc470fb0bd7659553d77fe965864d6734bbe6576f4` |
| Direct prior art | Mind-VLA | [prior-art-mind-vla-arxiv-v2.pdf](prior-art-mind-vla-arxiv-v2.pdf) | arXiv:2608.04633 v2 | 9 | 1,665,341 | `12a01a0f65c7331505e8c8a4c41eb356b56cb55173db9479c8dfc3ce4944bddb` |
| Adjacent | ReconVLA | [adjacent-reconvla-arxiv-v1.pdf](adjacent-reconvla-arxiv-v1.pdf) | arXiv:2508.10333 v1; AAAI 2026 venue identity separately confirmed | 10 | 33,378,540 | `de568f5eb3e26a355160e7b79c286327eae38f3585a419a51a061c186e55f011` |

## Acquisition records

| Work | Primary record / acquisition source |
|---|---|
| REPA | [ICLR 2025 record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf) · [arXiv record](https://arxiv.org/abs/2410.06940) · [code](https://github.com/sihyun-yu/REPA) |
| iREPA | [ICLR 2026 record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/3929a7785bd56f57edcff0152ab41289-Paper-Conference.pdf) · [arXiv record](https://arxiv.org/abs/2512.10794) · [code](https://github.com/end2end-diffusion/irepa) |
| FLARE | [PMLR record](https://proceedings.mlr.press/v305/zheng25a.html) · [PMLR PDF](https://proceedings.mlr.press/v305/zheng25a/zheng25a.pdf) |
| Spatial Forcing | [arXiv v2](https://arxiv.org/abs/2510.12276v2) · [project](https://spatial-forcing.github.io/) |
| FRAPPE | [arXiv v1](https://arxiv.org/abs/2602.17259v1) · [code](https://github.com/Jbo-Wang/frappe) |
| FutureVLA | [arXiv v1](https://arxiv.org/abs/2603.10712v1) |
| VEGA | [arXiv v1](https://arxiv.org/abs/2605.10485v1) |
| AGRA | [arXiv v1](https://arxiv.org/abs/2606.12217v1) · [project](https://xpeng-robotics.github.io/agra) |
| SAM3D-Guided | [arXiv v1](https://arxiv.org/abs/2607.25912v1) |
| Robust-WAM | [arXiv v1](https://arxiv.org/abs/2608.05903v1) |
| Mind-VLA | [arXiv v2](https://arxiv.org/abs/2608.04633v2) |
| ReconVLA | [arXiv v1](https://arxiv.org/abs/2508.10333v1) · [AAAI record](https://doi.org/10.1609/aaai.v40i22.38921) |

## Version cautions

- `REPA`: the local canonical artifact is the ICLR 2025 final. arXiv currently has later revision metadata; do not silently mix revised arXiv text with venue-final page locators.
- `iREPA`: the local canonical artifact is the ICLR 2026 final.
- `Spatial Forcing`: acceptance status can be cited from its venue record, but page locators in this package refer to arXiv v2.
- `ReconVLA`: the AAAI 2026 DOI establishes venue identity; the local full text remains arXiv v1 because the venue download did not complete successfully.
- All 2026 arXiv-only candidates are preprints at this snapshot unless a venue record is explicitly listed. Their results require the usual reproducibility and version-drift caveat.

## Leftover-file check

No partial `.part`, `.tmp`, or zero-byte PDF was retained in this companion directory. The `validation/` subdirectory is reserved for future machine-readable checks and is not a paper source.
