from __future__ import annotations

import csv
from pathlib import Path

# ----------------------------------------------------------------------------
# Assignment: two-feature linear regression   expenses = w0 + w1*bmi + w2*age
# Builds on script 02 (Normal Equation) and script 03 (Gradient Descent).
#
# Works whether this file sits next to the data/ folder (like scripts 01-04)
# or inside src/ (data/ one level up).
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
KAGGLE_URL = "https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction"

FEATURES = ["bmi", "age"]
BASELINE_FEATURE = "bmi"
TARGET = "expenses"

LEARNING_RATE = 0.05
EPOCHS = 10000
LOG_EVERY = 1000


# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def find_project_root_and_csv() -> tuple[Path, Path]:
    """Look for insurance.csv the way the course scripts expect it, plus a few fallbacks."""
    tried: list[Path] = []
    for root in (SCRIPT_DIR, SCRIPT_DIR.parent):
        for rel in (
            Path("data") / "insurance-premium-prediction" / "insurance.csv",
            Path("data") / "insurance.csv",
            Path("insurance.csv"),
        ):
            candidate = root / rel
            tried.append(candidate)
            if candidate.exists():
                return root, candidate

    lines = "\n".join(f"  - {p}" for p in tried)
    raise SystemExit(
        "\ninsurance.csv was not found. Looked in:\n"
        f"{lines}\n\n"
        "Download the dataset from Kaggle:\n"
        f"  {KAGGLE_URL}\n\n"
        "Then save the file as:\n"
        "  data/insurance-premium-prediction/insurance.csv\n"
        "(next to the course scripts) and run this script again.\n"
    )


def load_xy(csv_path: Path, features: list[str], target: str) -> tuple[list[list[float]], list[float]]:
    """Return rows of feature values (one list per row) and the target list."""
    X: list[list[float]] = []
    y: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        needed = features + [target]
        missing = [c for c in needed if c not in columns]
        if missing:
            raise SystemExit(f"Missing column(s) {missing} in {csv_path}. Found: {columns}")
        for row in reader:
            X.append([float(row[name]) for name in features])
            y.append(float(row[target]))
    return X, y


def column(X: list[list[float]], j: int) -> list[float]:
    return [row[j] for row in X]


# ----------------------------------------------------------------------------
# Metrics (mse / r2_score are the same helpers as scripts 02 and 03)
# ----------------------------------------------------------------------------
def mse(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return mse(y_true, y_pred) ** 0.5


def mae(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    return 1.0 - (ss_res / ss_tot if ss_tot else 0.0)


def predict(X: list[list[float]], w: list[float]) -> list[float]:
    """w = [w0, w1, w2, ...]; each row of X holds the feature values in the same order."""
    return [w[0] + sum(wj * xj for wj, xj in zip(w[1:], row)) for row in X]


# ----------------------------------------------------------------------------
# Baseline: single-feature Normal Equation from script 02 (2x2 shortcut)
# ----------------------------------------------------------------------------
def fit_normal_equation_single_feature(x: list[float], y: list[float]) -> tuple[float, float]:
    """Fit y = w0 + w1*x using theta=(X^T X)^-1 X^T y for one feature (script 02)."""
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(v * v for v in x)
    sum_xy = sum(vx * vy for vx, vy in zip(x, y))

    a, b, c, d = float(n), sum_x, sum_x, sum_x2
    det = a * d - b * c
    if det == 0:
        raise ValueError("Singular matrix in normal equation; cannot invert X^T X.")

    inv_xtx = [[d / det, -b / det], [-c / det, a / det]]
    xty = [sum_y, sum_xy]
    w0 = inv_xtx[0][0] * xty[0] + inv_xtx[0][1] * xty[1]
    w1 = inv_xtx[1][0] * xty[0] + inv_xtx[1][1] * xty[1]
    return w0, w1


# ----------------------------------------------------------------------------
# 2. Normal Equation for ANY number of features (general matrix solve)
#    theta = (X^T X)^-1 X^T y  ->  solve (X^T X) theta = X^T y by Gauss-Jordan.
# ----------------------------------------------------------------------------
def solve_gauss_jordan(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b using Gauss-Jordan elimination with partial pivoting."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]  # augmented matrix [A | b]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix: features are linearly dependent.")
        M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        M[col] = [v / pivot for v in M[col]]

        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]

    return [M[i][n] for i in range(n)]


def fit_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    """Return [w0, w1, ..., wk] for k features."""
    Xb = [[1.0] + row for row in X]  # add a column of 1s for the intercept w0
    size = len(Xb[0])

    xtx = [[sum(r[i] * r[j] for r in Xb) for j in range(size)] for i in range(size)]
    xty = [sum(r[i] * yi for r, yi in zip(Xb, y)) for i in range(size)]
    return solve_gauss_jordan(xtx, xty)


# ----------------------------------------------------------------------------
# 3. Gradient Descent with standardized features (extends script 03)
# ----------------------------------------------------------------------------
def standardize(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0 for _ in values], mean, 1.0
    return [(v - mean) / std for v in values], mean, std


def fit_gradient_descent(X: list[list[float]], y: list[float]) -> list[float]:
    """Full-batch gradient descent on standardized features; returns weights in ORIGINAL units."""
    n_features = len(X[0])
    n = len(y)

    # Standardize each feature column separately, remember mean/std to convert back later.
    cols_std: list[list[float]] = []
    means: list[float] = []
    stds: list[float] = []
    for j in range(n_features):
        z, m, s = standardize(column(X, j))
        cols_std.append(z)
        means.append(m)
        stds.append(s)
    Z = [[cols_std[j][i] for j in range(n_features)] for i in range(n)]  # back to row form

    w0 = 0.0
    w = [0.0] * n_features  # weights in standardized space

    print("Gradient Descent training (standardized bmi + age):")
    for epoch in range(1, EPOCHS + 1):
        preds_std = [w0 + sum(wj * zj for wj, zj in zip(w, row)) for row in Z]
        errors = [p - yi for p, yi in zip(preds_std, y)]

        grad_w0 = (2.0 / n) * sum(errors)
        grad_w = [(2.0 / n) * sum(err * row[j] for err, row in zip(errors, Z)) for j in range(n_features)]

        w0 -= LEARNING_RATE * grad_w0
        w = [wj - LEARNING_RATE * g for wj, g in zip(w, grad_w)]

        if epoch % LOG_EVERY == 0 or epoch == 1:
            current_loss = mse(y, [w0 + sum(wj * zj for wj, zj in zip(w, row)) for row in Z])
            print(f"  epoch={epoch:5d} mse={current_loss:.2f}")

    # Convert back from standardized space to original units:
    #   y = w0 + sum(w_j * (x_j - mean_j) / std_j)
    #     = (w0 - sum(w_j * mean_j / std_j)) + sum((w_j / std_j) * x_j)
    w_original = [w[j] / stds[j] for j in range(n_features)]
    w0_original = w0 - sum(w[j] * means[j] / stds[j] for j in range(n_features))
    return [w0_original] + w_original


# ----------------------------------------------------------------------------
# 5/6. Comparison table + CSV export
# ----------------------------------------------------------------------------
def build_row(name: str, w: list[float], X: list[list[float]], y: list[float]) -> dict[str, float | str]:
    preds = predict(X, w)
    return {
        "model": name,
        "w0": w[0],
        "w1_bmi": w[1],
        "w2_age": w[2] if len(w) > 2 else "",
        "MSE": mse(y, preds),
        "RMSE": rmse(y, preds),
        "MAE": mae(y, preds),
        "R2": r2_score(y, preds),
    }


def fmt(value: float | str, spec: str) -> str:
    return "-" if value == "" else format(value, spec)


def print_comparison_table(rows: list[dict[str, float | str]]) -> None:
    headers = ["model", "w0", "w1_bmi", "w2_age", "MSE", "RMSE", "MAE", "R2"]
    specs = [None, ",.2f", ",.2f", ",.2f", ",.0f", ",.2f", ",.2f", ".4f"]
    body = [
        [str(r["model"])] + [fmt(r[h], s) for h, s in zip(headers[1:], specs[1:])]
        for r in rows
    ]
    widths = [max(len(h), *(len(line[i]) for line in body)) for i, h in enumerate(headers)]

    def render(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    header_line = render(headers)
    print(header_line)
    print("-" * len(header_line))
    for line in body:
        print(render(line))


def write_results_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (v if isinstance(v, str) else round(v, 4)) for k, v in r.items()})


def print_pipeline_outline() -> None:
    print("Pipeline outline (assignment):")
    print("  1) Read data: load bmi, age (features) and expenses (target).")
    print("  2) Normal Equation: solve for w0, w1, w2 with a general matrix solve.")
    print("  3) Gradient Descent: standardize both features, train, convert weights back.")
    print("  4) Evaluate: MSE, RMSE, MAE, R^2 for both models.")
    print("  5) Compare against the bmi-only baseline from script 02.")
    print("  6) Export the comparison table to reports/assignment_results.csv.")
    print()


def main() -> None:
    print_pipeline_outline()

    project_root, csv_path = find_project_root_and_csv()
    X, y = load_xy(csv_path, FEATURES, TARGET)
    X_baseline = [[v] for v in column(X, FEATURES.index(BASELINE_FEATURE))]

    print(f"Dataset: {csv_path}  ({len(y)} rows)")
    print(f"Model:   {TARGET} = w0 + w1*bmi + w2*age")
    print()

    # Baseline (script 02 method, bmi only)
    b0, b1 = fit_normal_equation_single_feature(column(X, 0), y)

    # Two-feature models
    w_ne = fit_normal_equation(X, y)
    w_gd = fit_gradient_descent(X, y)
    print()

    rows = [
        build_row("Baseline (bmi only)", [b0, b1], X_baseline, y),
        build_row("Normal Equation (bmi + age)", w_ne, X, y),
        build_row("Gradient Descent (bmi + age)", w_gd, X, y),
    ]

    print("Comparison table:")
    print_comparison_table(rows)
    print()

    gain = float(rows[1]["R2"]) - float(rows[0]["R2"])
    max_diff = max(abs(a - b) for a, b in zip(w_ne, w_gd))
    print(f"R^2 gain from adding age (Normal Equation vs baseline): {gain:+.4f}")
    print(f"Largest weight difference, Normal Equation vs Gradient Descent: {max_diff:.6f}")
    print()

    print("What you are looking at:")
    print("  w1 (bmi): change in predicted expenses per +1 BMI, holding age fixed.")
    print("  w2 (age): change in predicted expenses per +1 year, holding BMI fixed.")
    print("  RMSE/MAE are in dollars; MSE is in squared dollars; R^2 is the share of variation explained.")
    print("  Both methods minimize the same squared error, so their weights should match")
    print("  (a small difference means Gradient Descent has not fully converged).")

    out_path = project_root / "reports" / "assignment_results.csv"
    write_results_csv(rows, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()