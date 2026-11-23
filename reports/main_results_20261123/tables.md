# Computed tables

Coverage is intersection / goal area, not success or IoU.

## Test: mean over 16k / 18k / 20k

| Input | Goal | Demos | Method | Coverage | Seed SD | Success | 20k only | Limited grid max |
|---|---|---:|---|---:|---:|---:|---:|---:|
| image | fixed | 50 | dp | 0.3241 | 0.0549 | 6.44% | 0.3244 | 0.3287 |
| image | fixed | 50 | equidiff_c4 | 0.3348 | 0.0073 | 1.33% | 0.3336 | 0.3448 |
| image | fixed | 50 | equidiff_c4_gate | 0.3087 | 0.0310 | 1.11% | 0.3206 | 0.3263 |
| image | rotated | 25 | dp | 0.2076 | 0.0126 | 0.11% | 0.2112 | 0.2122 |
| image | rotated | 25 | dp_c4_aug | 0.4598 | 0.0604 | 2.00% | 0.4526 | 0.4754 |
| image | rotated | 25 | equidiff_c4 | 0.1893 | 0.0236 | 0.33% | 0.1831 | 0.1972 |
| image | rotated | 50 | dp | 0.2899 | 0.0385 | 1.00% | 0.2916 | 0.2985 |
| image | rotated | 50 | dp_c4_aug | 0.5787 | 0.0104 | 3.78% | 0.5796 | 0.5854 |
| image | rotated | 50 | equidiff_c4 | 0.3465 | 0.0173 | 2.33% | 0.3670 | 0.3670 |
| image | rotated | 50 | equidiff_c4_gate | 0.3225 | 0.0497 | 2.44% | 0.3289 | 0.3330 |
| image | rotated | 50 | equidiff_c8 | 0.2903 | 0.0388 | 0.44% | 0.2990 | 0.3003 |
| image | rotated | 100 | dp | 0.4675 | 0.0460 | 1.33% | 0.4715 | 0.4763 |
| image | rotated | 100 | dp_c4_aug | 0.6166 | 0.0194 | 1.11% | 0.6333 | 0.6342 |
| image | rotated | 100 | equidiff_c4 | 0.5789 | 0.0277 | 5.89% | 0.5999 | 0.5999 |
| image | rotated | 200 | dp | 0.5721 | 0.0276 | 2.67% | 0.5819 | 0.5892 |
| image | rotated | 200 | dp_c4_aug | 0.5828 | 0.0074 | 0.78% | 0.6031 | 0.6049 |
| image | rotated | 200 | equidiff_c4 | 0.6712 | 0.0518 | 11.89% | 0.6802 | 0.6854 |
| image | rotated | 200 | equidiff_c8 | 0.5826 | 0.0545 | 7.56% | 0.5902 | 0.5973 |
| state | fixed | 50 | state_dp | 0.7869 | 0.0270 | 34.89% | 0.7874 | 0.7938 |
| state | fixed | 50 | state_equidiff | 0.4888 | 0.0232 | 5.67% | 0.4849 | 0.5099 |
| state | fixed | 50 | state_equidiff_gate | 0.6223 | 0.0075 | 18.89% | 0.6400 | 0.6483 |
| state | fixed | 50 | state_equidiff_goal | 0.6183 | 0.0340 | 17.00% | 0.6244 | 0.6338 |
| state | rotated | 50 | state_dp | 0.4061 | 0.0123 | 1.89% | 0.4070 | 0.4207 |
| state | rotated | 50 | state_equidiff | 0.4733 | 0.0340 | 6.67% | 0.4752 | 0.4888 |
| state | rotated | 50 | state_equidiff_gate | 0.4771 | 0.0492 | 4.00% | 0.4666 | 0.4851 |
| state | rotated | 50 | state_equidiff_goal | 0.5584 | 0.0085 | 19.33% | 0.5528 | 0.5705 |

## Independent validation curves

| Demos | Method | 20k | 40k | 60k |
|---:|---|---:|---:|---:|
| 25 | dp | 0.1646 | 0.1664 | 0.1661 |
| 25 | dp_c4_aug | 0.4215 | 0.4442 | 0.4700 |
| 25 | equidiff_c4 | 0.1716 | 0.1961 | 0.1945 |
| 50 | dp | 0.2254 | 0.2304 | 0.2310 |
| 50 | dp_c4_aug | 0.6278 | 0.6335 | 0.6377 |
| 50 | equidiff_c4 | 0.3181 | 0.3677 | 0.4053 |
| 100 | dp | 0.4372 | 0.4440 | 0.4551 |
| 100 | dp_c4_aug | 0.6472 | 0.7342 | 0.7702 |
| 100 | equidiff_c4 | 0.5322 | 0.5888 | 0.6260 |
| 200 | dp | 0.5765 | 0.6010 | 0.5964 |
| 200 | dp_c4_aug | 0.5751 | 0.6891 | 0.7422 |
| 200 | equidiff_c4 | 0.6386 | 0.7304 | 0.7501 |

## Paired coverage differences (left minus right)

| Input | Goal | Demos | Left | Right | Difference | 95% CI |
|---|---|---:|---|---|---:|---|
| image | fixed | 50 | dp | equidiff_c4 | -0.0107 | [-0.0876, 0.0653] |
| image | fixed | 50 | dp | equidiff_c4_gate | 0.0154 | [-0.0759, 0.1012] |
| image | fixed | 50 | equidiff_c4 | equidiff_c4_gate | 0.0261 | [-0.0255, 0.0702] |
| image | rotated | 25 | dp | dp_c4_aug | -0.2522 | [-0.3202, -0.1852] |
| image | rotated | 25 | dp | equidiff_c4 | 0.0183 | [-0.0262, 0.0597] |
| image | rotated | 25 | dp_c4_aug | equidiff_c4 | 0.2705 | [0.1871, 0.3597] |
| image | rotated | 50 | dp | dp_c4_aug | -0.2887 | [-0.3573, -0.2178] |
| image | rotated | 50 | dp | equidiff_c4 | -0.0565 | [-0.1391, 0.0220] |
| image | rotated | 50 | dp | equidiff_c4_gate | -0.0326 | [-0.1014, 0.0348] |
| image | rotated | 50 | dp | equidiff_c8 | -0.0003 | [-0.0888, 0.0779] |
| image | rotated | 50 | dp_c4_aug | equidiff_c4 | 0.2322 | [0.1512, 0.3118] |
| image | rotated | 50 | dp_c4_aug | equidiff_c4_gate | 0.2562 | [0.1616, 0.3466] |
| image | rotated | 50 | dp_c4_aug | equidiff_c8 | 0.2884 | [0.1945, 0.3733] |
| image | rotated | 50 | equidiff_c4 | equidiff_c4_gate | 0.0239 | [-0.0516, 0.0920] |
| image | rotated | 50 | equidiff_c4 | equidiff_c8 | 0.0562 | [0.0151, 0.1048] |
| image | rotated | 50 | equidiff_c4_gate | equidiff_c8 | 0.0323 | [-0.0335, 0.1127] |
| image | rotated | 100 | dp | dp_c4_aug | -0.1492 | [-0.2065, -0.1005] |
| image | rotated | 100 | dp | equidiff_c4 | -0.1115 | [-0.1896, -0.0279] |
| image | rotated | 100 | dp_c4_aug | equidiff_c4 | 0.0377 | [-0.0260, 0.1145] |
| image | rotated | 200 | dp | dp_c4_aug | -0.0107 | [-0.0608, 0.0289] |
| image | rotated | 200 | dp | equidiff_c4 | -0.0991 | [-0.1596, -0.0247] |
| image | rotated | 200 | dp | equidiff_c8 | -0.0106 | [-0.0979, 0.0782] |
| image | rotated | 200 | dp_c4_aug | equidiff_c4 | -0.0884 | [-0.1452, -0.0193] |
| image | rotated | 200 | dp_c4_aug | equidiff_c8 | 0.0001 | [-0.0657, 0.0778] |
| image | rotated | 200 | equidiff_c4 | equidiff_c8 | 0.0885 | [0.0223, 0.1621] |
| state | fixed | 50 | state_dp | state_equidiff | 0.2981 | [0.2282, 0.3676] |
| state | fixed | 50 | state_dp | state_equidiff_gate | 0.1646 | [0.1070, 0.2248] |
| state | fixed | 50 | state_dp | state_equidiff_goal | 0.1686 | [0.0932, 0.2448] |
| state | fixed | 50 | state_equidiff | state_equidiff_gate | -0.1335 | [-0.1938, -0.0732] |
| state | fixed | 50 | state_equidiff | state_equidiff_goal | -0.1295 | [-0.1915, -0.0631] |
| state | fixed | 50 | state_equidiff_gate | state_equidiff_goal | 0.0039 | [-0.0543, 0.0570] |
| state | rotated | 50 | state_dp | state_equidiff | -0.0672 | [-0.1462, 0.0149] |
| state | rotated | 50 | state_dp | state_equidiff_gate | -0.0709 | [-0.1346, -0.0165] |
| state | rotated | 50 | state_dp | state_equidiff_goal | -0.1522 | [-0.2582, -0.0416] |
| state | rotated | 50 | state_equidiff | state_equidiff_gate | -0.0038 | [-0.1045, 0.0822] |
| state | rotated | 50 | state_equidiff | state_equidiff_goal | -0.0851 | [-0.1880, 0.0203] |
| state | rotated | 50 | state_equidiff_gate | state_equidiff_goal | -0.0813 | [-0.1924, 0.0341] |
