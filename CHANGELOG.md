# Changelog

## [1.3.0] - 2026-04-25

### Added — Model Section (20 widgets)

This release delivers the full **Model** section of Portakal, implementing all 20 Orange-equivalent
learner widgets using scikit-learn as the computation backend.

#### New Dependencies
- `scikit-learn >= 1.4`
- `scipy >= 1.12`
- `numpy >= 1.26`

#### New Widgets

| Widget | Output Port | Notes |
|---|---|---|
| **Constant** | Model | `DummyClassifier` / `DummyRegressor` — most-frequent or mean baseline |
| **Naive Bayes** | Classifier | `GaussianNB` — classification only |
| **kNN** | Model | k-Nearest Neighbours; Euclidean / Manhattan / Chebyshev metric; uniform or distance-weighted |
| **Tree** | Tree | Decision Tree with binary split, min-leaf, min-internal, max-depth and majority-stop controls; outputs `DecisionTreeArtifact` for Tree Viewer / Pythagorean Tree |
| **Random Forest** | Random Forest | Ensemble of decision trees; outputs `RandomForestArtifact` for Pythagorean Forest |
| **Logistic Regression** | Classifier | L1 / L2 / no regularisation, C strength slider, balance class; outputs `LogisticRegressionClassifierArtifact` for Nomogram |
| **Linear Regression** | Model | OLS / Ridge / Lasso / Elastic Net with alpha and L1-ratio sliders |
| **SVM** | Model | C-SVM and ν-SVM; Linear / Polynomial / RBF / Sigmoid kernels; per-kernel g/c/d parameters |
| **Neural Network** | Model | MLP; hidden layers text field; Identity / Logistic / tanh / ReLU activation; L-BFGS-B / SGD / Adam solver; alpha regularisation slider |
| **AdaBoost** | Model | `AdaBoostClassifier` / `AdaBoostRegressor`; n_estimators, learning_rate, regression loss, optional fixed seed |
| **Gradient Boosting** | Model | sklearn `GradientBoosting`; n_estimators, learning_rate, lambda slider, max_depth, min_samples_split, subsample |
| **Stochastic Gradient Descent** | Model | Full Orange-equivalent UI: classification and regression loss combos with ε, regularisation type, strength and L1-ratio, three learning rate schedules, η₀, power_t, shuffle and seed |
| **PLS** | Model | `PLSRegression`; n_components, max_iter, scale toggle; numeric target required |
| **Curve Fit** | Model | User-defined expression + dynamic parameter table (name, initial value, optional lower/upper bounds); fitted with `scipy.optimize.curve_fit` |
| **Scoring Sheet** | Classifier | Fast point-based explainable classifier; uses existing `ScoringSheetService`; outputs `ScoringSheetClassifierArtifact` for Scoring Sheet Viewer |
| **CN2 Rule Induction** | Classifier | Orange CN2-style rule inducer; uses existing `CN2RuleInductionService`; outputs `CN2RuleClassifierArtifact` for CN2 Rule Viewer |
| **Calibrated Learner** | Classifier | Wraps any sklearn-backed classifier with `CalibratedClassifierCV`; Sigmoid / Isotonic / no calibration; CA / F1 / no threshold optimisation |
| **Stacking** | Model | `StackingClassifier` / `StackingRegressor`; accepts multiple Model inputs and an optional Aggregate meta-learner |
| **Save Model** | *(none)* | Pickles any model artifact to a `.portakal` file |
| **Load Model** | Model | Loads a pickled `.portakal` artifact and emits it downstream |

#### New Infrastructure

- `src/portakal_app/sklearn_model_artifacts.py` — `SklearnModelArtifact` dataclass holding the unfitted estimator, the fitted model, encoding maps, and training metadata
- `src/portakal_app/data/services/sklearn_learner_service.py` — Universal encoding + training service (one-hot for categoricals, mean impute for numerics) that produces `SklearnModelArtifact`
- `src/portakal_app/data/services/tree_service.py` — Converts a fitted sklearn `DecisionTreeClassifier` / `DecisionTreeRegressor` into a `DecisionTreeArtifact` (compatible with Tree Viewer and Pythagorean Tree)
- `src/portakal_app/data/services/random_forest_service.py` — Wraps the tree service per estimator to produce a `RandomForestArtifact` (compatible with Pythagorean Forest)
- `src/portakal_app/ui/screens/model_base.py` — Shared scaffold (`ModelScreenBase`) for all learner screens: dataset label, status line, auto-apply checkbox, Apply button

#### Port Compatibility Overrides

New entries in `WORKFLOW_PORT_COMPATIBILITY_OVERRIDES` (`models.py`):

- `("Tree", "Model")` → `save-model`, `stacking`
- `("Classifier", "Model")` → `save-model`, `stacking`
- `("Random Forest", "Model")` → `save-model`, `stacking`
- `("Model", "Classifier")` → `cn2-rule-viewer`, `nomogram`, `scoring-sheet-viewer`, `calibrated-learner`
- `("Tree/Classifier/Random Forest", "Aggregate")` → `stacking`

---

## [1.2.0] — Visualize Section

Added 18 Orange-equivalent Visualize widgets:
Box Plot, Violin Plot, Distributions, Scatter Plot, Line Plot, Bar Plot,
Sieve Diagram, Mosaic Display, FreeViz, Linear Projection, Radviz, Heat Map,
Venn Diagram, Silhouette Plot, Pythagorean Forest, Pythagorean Tree,
CN2 Rule Viewer, Nomogram, Scoring Sheet Viewer.

---

## [1.1.0] — Transform Section

Added 24 Orange-equivalent Transform widgets:
Select by Data Index, Randomize, Purge Domain, Unique, Apply Domain,
Data Sampler, Select Columns, Select Rows, Transpose, Split, Merge Data,
Concatenate, Aggregate Columns, Group By, Pivot Table, Preprocess,
Impute, Continuize, Discretize, Melt, Create Class, Create Instance,
Formula, Python Script.

---

## [1.0.0] — Data Section & Shell

Initial release.

- PySide6 application shell with category sidebar, widget catalog, and workflow canvas
- 11 active Data widgets: File, CSV File Import, Datasets, Data Table, Paint Data,
  Data Info, Rank, Edit Domain, Color, Column Statistics, Save Data
- Workflow engine: typed port connections, payload propagation, node state
  serialisation/deserialisation
