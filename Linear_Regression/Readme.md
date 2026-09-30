# Two-Feature Linear Regression Extension

Extends the single-feature model from Module 2 (`expenses = w0 + w1*bmi`) to a two-feature model:

```
expenses = w0 + w1*bmi + w2*age
```

The model is fit two ways and compared against the `bmi`-only baseline from `02_ols_normal_equation.py`:

1. **Normal Equation** with a general matrix solve (Gauss-Jordan elimination), not the 2x2 shortcut.
2. **Gradient Descent** on standardized features, with weights converted back to original units.

---

## Project structure

```
Linear_Regression/
├── data/
│   └── insurance.csv
├── reports/
│   └── assignment_results.csv      (created when you run the script)
├── Src/
│   └── Two_feature_model.py
├── 02_ols_normal_equation.py       (course script: bmi-only Normal Equation)
├── 03_gradient_descent.py          (course script: bmi-only Gradient Descent)
├── WRITTEN MEMO                       (written memo)
├── requirements.txt
└── README.md

---

## 1. Prerequisites

| Tool | Needed? | Notes |
|---|---|---|
| Python 3.9 or newer | Yes | Check with `python --version` (or `python3 --version`) |
| Git | Only to clone | Or download the repo as a ZIP from GitHub |
| Python packages | **No** | The script uses the standard library only (`csv`, `pathlib`) |
| Kaggle account | Only to download the dataset | Free |

---

## 2. Clone the repository

```bash
git clone https://github.com/Vaishnaviram1026/Machine_Learning.git
cd Machine_Learning/Linear_Regression
```

No Git? On the repo page choose **Code -> Download ZIP**, unzip it, and open a terminal in the `Linear_Regression` folder.

---

## 3. Virtual environment (optional)

The script needs no third-party packages, so this step can be skipped. To keep your setup clean anyway:

**Windows (PowerShell)**
```powershell
python -m venv linearvenv
linearvenv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process RemoteSigned` first, or use Command Prompt with `linearvenv\Scripts\activate.bat`.

**Mac / Linux**
```bash
python3 -m venv linearvenv
source linearvenv/bin/activate
```

Then install the (optional) requirements:

```bash
pip install -r requirements.txt
```

The environment folder `linearvenv/` is listed in `.gitignore` and is not part of the repository. Run `deactivate` to leave the environment.

---

## 4. Get the dataset

Download `insurance.csv` from Kaggle:
https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction

Log in, click **Download**, unzip if needed, and save the file as:

```
Linear_Regression/data/insurance.csv
```

The script also finds `data/insurance-premium-prediction/insurance.csv`. If it can't find the file, it prints every path it checked plus the Kaggle link.

Required columns: `bmi`, `age`, `expenses`.

---

## 5. Run

From inside the `Linear_Regression` folder (with the virtual environment active, if you made one):

```bash
python Src/assignment_two_feature_model.py
```

Use `python3` on Mac/Linux if `python` is not found, or `py` on Windows. Run it from a terminal, not by double-clicking the file.

You should see the training log, a comparison table, and `Saved: .../reports/assignment_results.csv`.

---

## 6. What the script does

1. Loads `bmi`, `age` and `expenses` from `insurance.csv`.
2. Fits the two-feature model with the Normal Equation (`fit_normal_equation`), which builds XᵀX and Xᵀy for any number of features and solves them with `solve_gauss_jordan`.
3. Fits the same model with Gradient Descent (`fit_gradient_descent`): standardizes each feature, trains (learning rate 0.05, 10,000 epochs, as in script 03), then converts the weights back to original units.
4. Computes MSE, RMSE, MAE and R² for each model.
5. Compares both two-feature models with the `bmi`-only baseline from script 02.
6. Prints a comparison table and writes it to `reports/assignment_results.csv`.

---

## 7. Output

`reports/assignment_results.csv` columns: `model, w0, w1_bmi, w2_age, MSE, RMSE, MAE, R2`.
The baseline row leaves `w2_age` empty because that model has no age feature.

Expected results:

| Model | w0 | w1 (bmi) | w2 (age) | RMSE | MAE | R² |
|---|---|---|---|---|---|---|
| Baseline (bmi only) | 1178.18 | 394.33 | (blank) | 11,864.41 | 9,172.30 | 0.0394 |
| Normal Equation (bmi + age) | -6437.35 | 333.39 | 241.90 | 11,373.64 | 9,032.28 | 0.1173 |
| Gradient Descent (bmi + age) | -6437.35 | 333.39 | 241.90 | 11,373.64 | 9,032.28 | 0.1173 |

Adding age raises R² by about 0.078, and both methods reach the same weights.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Install Python from python.org (tick "Add to PATH" on Windows), or try `python3` / `py` |
| `insurance.csv was not found` | Save the CSV as `data/insurance.csv` inside `Linear_Regression` |
| `Missing column(s) ... Found: [...]` | Rename the column in the CSV, or edit `TARGET` / `FEATURES` at the top of the script (some copies use `charges` instead of `expenses`) |
| Nothing prints when double-clicking the file | Run it from a terminal instead |
| Weights differ slightly between the two methods | Increase `EPOCHS` in the script; Gradient Descent needs to fully converge |

---

## Notes

- Nothing is hardcoded. All weights and metrics are computed from the CSV.
- The baseline intentionally uses the 2x2 single-feature formula from script 02, as the assignment requires.
- To use other features, change `FEATURES` at the top of the script. The two fitting functions accept any number of features, but the printed table and CSV headers are set up for `bmi` and `age`.

## Deliverables

1. `Src/Two_feature_model.py`
2. `reports/assignment_results.csv`
3. Written memo 