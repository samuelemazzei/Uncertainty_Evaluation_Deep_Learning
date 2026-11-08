# Clinical Deep Learning Under Uncertainty

Evaluating predictive reliability in multi-label risk estimation using generative synthetic patients.

This repository contains the code of my master's dissertation in Business Analytics (University of Bologna, academic year 2025-2026). It asks a simple question. Can a clinical deep learning model be trusted when it meets data unlike anything seen in training?

Using survey and biomarker data from SHARE, the project predicts four diseases jointly: diabetes, stroke, heart attack and cancer. Five multi-label models are compared. Two of them report their own uncertainty: a Bayesian neural network (BNN) and an MLP with Monte Carlo Dropout. Both are then stress-tested on synthetic patients generated with CTGAN and TVAE, under gradual noise, extreme out-of-distribution values and feature permutation.

## Key findings

1. **Both uncertainty-aware models are overconfident before any shift.** Their reliability curves sit below the diagonal for all four diseases, with an expected calibration error (ECE) between 0.2 and 0.3.
2. **Level 1, noise injection.** MLP Dropout uncertainty rises with the noise level. BNN uncertainty stays flat or moves in small, non-monotonic steps. The BNN detects the shift only weakly through its epistemic channel.
3. **Level 2, extreme OOD probe.** Shifting every continuous feature 15% beyond the observed maximum pushes the predicted probability toward 1 for stroke and cancer in both models. MLP Dropout keeps a high ambiguity. The BNN saturates and its uncertainty collapses under both generators. It becomes more confident where it should not.
4. **Level 3, feature permutation.** Age dominates the response of both models under TVAE. Cystatin C matters more under CTGAN for cancer and stroke.
5. **No single ranking.** Noise injection favours MLP Dropout, and the OOD probe exposes the BNN collapse. Which model looks safer depends on the type of shift.
6. **Aggregate trends can hide individual behaviour.** Patient 1810, from the real test set, is misdiagnosed for cancer by both models. The BNN is confidently wrong and MLP Dropout is uncertainly wrong. Synthetic patient 604 moves against the population trend under noise injection.

The thesis argues that stress testing under distribution shift should be a required step before a clinical model is considered for real use, in addition to accuracy on a held-out test set.

### Level 1: noise injection

Uncertainty as a function of the noise level, for each disease and model.
### Level 2: extreme OOD probe

Mean uncertainty at baseline (sigma = 0) and under the OOD probe, by disease and generator. For stroke and cancer, the BNN uncertainty collapses.

### Level 3: feature permutation

Change in uncertainty when a single feature is permuted, sorted by mean effect. Filled markers are significant effects.

## Data

The data come from SHARE (Survey of Health, Ageing and Retirement in Europe), Wave 6. This wave is the first to combine the usual questionnaires with dried blood spot (DBS) laboratory measurements.

| Item | Value |
|---|---|
| Respondents after cleaning | 23,057 |
| Features | 23 (13 categorical, 10 numeric) |
| Labels | diabetes, heart attack, stroke, cancer |

Label prevalence in the full dataset:

| Label | Positive cases |
|---|---|
| Diabetes | 14.27% |
| Heart attack | 11.89% |
| Cancer | 4.27% |
| Stroke | 3.34% |

Feature groups:

- **Symptoms:** depression, appetite, concentration, fatigue.
- **Lifestyle and demographics:** physical activity, alcohol, age, gender, BMI, maximum grip strength.
- **Medication use:** blood pressure, stomach, sleep, inflammation, other.
- **Blood biomarkers:** total cholesterol, CRP, triglycerides, cystatin C, HDL, hemoglobin, HbA1c. The blood-sample file also provides the `astar` indicator.

Respondents without a cholesterol measurement are dropped. The biomarkers are imputed with MICE using Bayesian ridge regression, and the five imputed datasets are averaged cell by cell. BMI and grip strength are filled with gender-specific medians and means, and categorical variables with the mode. Implausible values are winsorized, which for CRP means the lower 0.25% tail. The smoking variable is dropped because 95% of its values are missing.

**The data are not included.** SHARE microdata cannot be redistributed. Register at the SHARE website to download them, and follow its citation requirements.

## Methods

### Balancing

The split is multi-label stratified (60% training, 20% validation, 20% test) with `iterative-stratification`. It gives 13,834 training, 4,612 validation and 4,611 test respondents.

Two oversampling strategies are used on the training set. SMOTEENN is applied to each label separately. MLSMOTE is applied to all labels jointly. The SCUMBLE index of the training labels is 0.00483, so standard MLSMOTE is used. The number of synthetic samples is tuned on the mean imbalance ratio per label (IRLbl). The grid is 2,000, 2,500, 2,750, 2,900 and 3,500, and 2,500 gives the lowest mean IRLbl (1.168). After MLSMOTE, SCUMBLE falls to 0.0006. A positive-class weight in the loss reinforces the balancing in the neural models.

### Models

| Notebook | Model |
|---|---|
| `logistic_regression` | Single-label and multi-output logistic regression, regularization tuned by grid search |
| `multitask_logistic` | Multi-task logistic regression in PyTorch |
| `random_forest_thresholds` | Random forest with balanced class weights and three threshold rules (Youden, recall, cost) |
| `mlp` | Multi-task MLP |
| `mlp_dropout` | MLP with MC Dropout (100 stochastic forward passes at inference) |
| `bayesian_nn` | Bayesian neural network trained with Bayes by Backprop (Blitz library, ELBO loss) |

The neural models use Adam with learning rate 0.001, 1,000 epochs and 5 random restarts. Hyperparameters are chosen by misclassification rate on the validation set: the hidden width and the weight decay (the KL weight for the BNN), plus the dropout rate for MC Dropout. Probabilities are turned into decisions with the Youden index computed on the validation set.

The two uncertainty-aware models report two quantities:

- **Uncertainty (epistemic):** the standard deviation of the Monte Carlo predictions, in percent.
- **Ambiguity (aleatoric):** the predictive entropy.

### Generative models

CTGAN and TVAE (SDV library) are trained for 300 epochs on the features of the balanced training set. They never see the real labels. The setting `enforce_min_max_values=False` lets synthetic patients fall outside the observed range.

Four sample sizes (500, 1,000, 2,000 and 5,000) are compared with the real data through the SDV quality score, the KS complement, the mean Hellinger distance and a classifier two-sample test (C2ST). A spread score summarizes how dispersed each synthetic sample is. The stress test uses the 1,000-sample configuration of each generator, which is the most dispersed one.

Neither generator reproduces the real data closely. C2ST AUC lies between 0.784 and 0.799, where 0.5 would mean indistinguishable.

### Stress testing

| Level | Perturbation | Settings |
|---|---|---|
| 1. Noise injection | Gaussian noise on continuous features, scaled on the standard deviation of the real data | sigma = 0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00 |
| 2. OOD probe | All continuous features shifted jointly beyond the training maximum | margin = 0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50 of the feature range; single-feature diagnostics |
| 3. Feature permutation | Each feature permuted across patients to break its relation with the others | 10 repeats |

Both models are run with 100 Monte Carlo samples per patient. Results are reported at population level and for individual patients.

## Results on the real test set

Test set: 4,611 respondents.

| Model | Macro F1 | Macro AUC | Macro AUC-PR | Hamming loss |
|---|---|---|---|---|
| Random Forest | 0.2393 | 0.7037 | 0.2084 | 0.2983 |
| MT Logistic | 0.2468 | 0.7315 | 0.1993 | 0.3224 |
| MLP | 0.2537 | 0.7328 | 0.2163 | 0.3256 |
| MLP Dropout | 0.2496 | 0.7324 | 0.2168 | 0.4024 |
| BNN | 0.2467 | 0.7233 | 0.2109 | 0.3705 |

Per-label comparison of the two uncertainty-aware models, shown as BNN / MLP Dropout:

| Label | AUC-ROC | AUC-PR | Brier | ECE | Mean MC std (%) |
|---|---|---|---|---|---|
| Diabetes | 0.831 / 0.825 | 0.454 / 0.437 | 0.172 / 0.182 | 0.244 / 0.269 | 0.79 / 5.87 |
| Stroke | 0.723 / 0.733 | 0.084 / 0.084 | 0.101 / 0.137 | 0.214 / 0.285 | 1.12 / 5.32 |
| Heart attack | 0.732 / 0.751 | 0.238 / 0.273 | 0.205 / 0.201 | 0.295 / 0.303 | 0.71 / 4.54 |
| Cancer | 0.607 / 0.620 | 0.061 / 0.068 | 0.103 / 0.145 | 0.208 / 0.296 | 1.25 / 6.13 |

Macro metrics hide the weak performance on rare labels. Both models are trained with a class-weighted loss, so predicted probabilities are not expected to match the raw label prevalences.

## Limitations

- The smoking variable is excluded. It is a shared risk factor for heart attack, stroke and cancer, so the low separability of stroke and cancer should be read with this in mind.
- SMOTEENN was applied to the full training set before cross-validation of the logistic regression. This leaks information across folds, and the logistic regression metrics may be slightly optimistic.
- MLSMOTE preserves label co-occurrence at the cost of single-label precision. Single-label results are not directly comparable with models that ignore co-occurrence.
- Both generators remain distinguishable from the real data. Some of the stress-test responses may come from generator artifacts and not from the shifts themselves.
- Synthetic data is a tool for testing. It does not replace validation on real patients, and its privacy protection is not assumed.

## Repository structure

```
clinical-dl-stress-testing/
├── data/
│   ├── raw/              SHARE .dta files (not tracked)
│   ├── interim/          balanced training data (not tracked)
│   ├── processed/        cleaned dataset (not tracked)
│   └── synthetic/        stress-test bundle with synthetic data (not tracked)
├── models/               trained models and result files (not tracked)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_balancing.ipynb
│   ├── 03a_logistic_regression.ipynb
│   ├── 03b_multitask_logistic.ipynb
│   ├── 03c_random_forest_thresholds.ipynb
│   ├── 03d_mlp.ipynb
│   ├── 03e_mlp_dropout.ipynb
│   ├── 03f_bayesian_nn.ipynb
│   ├── 04_generative_ai.ipynb
│   └── 05_stress_testing.ipynb
├── reports/
│   ├── figures/          figures used in this README
│   └── tables/
└── src/                  reusable Python modules (mlsmote.py)
```

## Getting started

Python 3.11 was used. The main dependencies are:

```
numpy  pandas  scipy  scikit-learn  statsmodels  matplotlib  seaborn
torch  blitz-bnn  sdv  imbalanced-learn  iterative-stratification
```

`balancing` also needs `mlsmote.py` in `src/`. It is the public MLSMOTE implementation by @niteshsukhwani and is not redistributed here.

## Reproducibility

Splits, resampling and model training use seed 1234. Synthetic data generation and the stress test derive their seeds from a global seed of 123. The classes `BNN` and `MLP_MC` are redefined in `05_stress_testing` because `pickle` needs them in the namespace at load time.

## Acknowledgements

- SHARE-ERIC for the data. This project uses SHARE data and is not endorsed by SHARE.
- [SDV](https://github.com/sdv-dev/SDV) for CTGAN and TVAE.
- [Blitz](https://github.com/piEsposito/blitz-bayesian-deep-learning) for Bayesian layers.
- @niteshsukhwani for the MLSMOTE implementation.
