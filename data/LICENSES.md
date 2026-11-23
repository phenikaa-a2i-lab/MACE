# Dataset licenses

The inputs in this directory come from two public benchmarks. Check each license before redistributing the files or using them outside research.

| Files | Source | License |
|---|---|---|
| `dev.jsonl`, `pilot_usmle_10.jsonl` | MedQA, US 4-option development split (`questions/US/4_options/phrases_no_exclude_dev.jsonl`) | MIT (repository license) |
| `vm14k_clean.jsonl`, `pilot_100.jsonl`, `vm14k_sub2000.jsonl`, `vm14k_sub2000.manifest.json` | VM14K, first Vietnamese medical benchmark | CC BY-NC 4.0, research use only (project page notice) |

## MedQA

- Source: https://github.com/jind11/MedQA
- License: MIT, as shown in the repository's `LICENSE` file and by the GitHub license metadata (checked 2026-11-23). The license text is reproduced below as the MIT license requires.
- The MIT license covers the repository. The questions come from medical licensing exams, and the repository does not state separate terms for them.
- Citation: D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, P. Szolovits. *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* Applied Sciences, 11(14):6421, 2021. arXiv:2009.13081.

```text
MIT License

Copyright (c) 2022 Di Jin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VM14K

- Source: https://arxiv.org/abs/2506.01305 (paper) and https://venera-ai.github.io/VM14K (project page).
- License: the project page states "The data, code and checkpoint is intended and licensed for research use only" and "The dataset is CC BY NC 4.0 (allowing only non-commercial use)". Commercial use is not allowed.
- The project page's notice also refers to the license agreements of CLIP, LLaMA, Vicuna and GPT-4, which looks copied from a template, and it is the only license statement found at the source. Third-party Hugging Face mirrors give different licenses (one lists Apache-2.0); they are not used here. If the exact terms matter for the submission, confirm them with the VM14K authors.
- Citation: *VM14K: First Vietnamese Medical Benchmark*, arXiv:2506.01305.
- `vm14k_clean.jsonl` is a cleaned copy of the public questions and `vm14k_sub2000.jsonl` is a stratified 2,000-question subset made by `code/dataio/make_vm14k_subset.py`. Both are derived from VM14K and keep its license.
