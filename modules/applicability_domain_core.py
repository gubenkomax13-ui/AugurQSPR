
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ad_build_williams_plot_df(
    y_true,
    y_pred,
    leverage,
    h_star
):
    """
    Формирует таблицу Williams Plot.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    leverage = np.asarray(leverage, dtype=float)

    residuals = y_true - y_pred

    residual_std = np.std(residuals, ddof=1)

    if residual_std <= 0:
        residual_std = 1.0

    standardized_residuals = residuals / residual_std

    df = pd.DataFrame({
        "Experimental": y_true,
        "Predicted": y_pred,
        "Residual": residuals,
        "Standardized Residual": standardized_residuals,
        "Leverage h": leverage,
    })

    df["Outside AD"] = df["Leverage h"] > h_star

    df["High Residual"] = (
        np.abs(df["Standardized Residual"]) > 3
    )

    df["Critical Point"] = (
        df["Outside AD"]
        & df["High Residual"]
    )

    return df
 
import matplotlib.pyplot as plt


def ad_make_williams_plot(
    williams_df,
    h_star
):
    """
    Williams plot.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    normal = williams_df[
        ~williams_df["Critical Point"]
    ]

    critical = williams_df[
        williams_df["Critical Point"]
    ]

    ax.scatter(
        normal["Leverage h"],
        normal["Standardized Residual"],
        alpha=0.7,
        s=35,
        label="Нормальные точки"
    )

    if len(critical) > 0:
        ax.scatter(
            critical["Leverage h"],
            critical["Standardized Residual"],
            s=60,
            marker="x",
            label="Критические точки"
        )

    ax.axvline(
        h_star,
        linestyle="--",
        linewidth=1.5,
        label=f"h* = {h_star:.3f}"
    )

    ax.axhline(
        3,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        -3,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Leverage h")
    ax.set_ylabel("Standardized residual")

    ax.set_title(
        "Williams Plot"
    )

    ax.set_xlabel(
        "Leverage h"
    )

    ax.set_ylabel(
        "Стандартизованный остаток"
    )

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig 
 
 
def qspr_calculate_leverage_ad(X_train, X_query=None, desc_names=None):
    """
    Applicability Domain через leverage.

    X_train — матрица обучающих дескрипторов.
    X_query — матрица веществ, для которых считаем leverage.
              Если None, считаем для самой обучающей выборки.

    Возвращает:
    {
        "leverage": ndarray,
        "threshold": float,
        "p": int,
        "n": int,
        "status": list[str]
    }
    """
    X_train = np.asarray(X_train, dtype=float)

    if X_query is None:
        X_query = X_train
    else:
        X_query = np.asarray(X_query, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train должен быть двумерной матрицей.")

    if X_query.ndim != 2:
        raise ValueError("X_query должен быть двумерной матрицей.")

    n, p = X_train.shape

    if n < 2:
        raise ValueError("Для Applicability Domain нужно минимум 2 вещества.")

    if X_query.shape[1] != p:
        raise ValueError(
            f"Число дескрипторов X_query ({X_query.shape[1]}) "
            f"не совпадает с X_train ({p})."
        )

    # Добавляем свободный член, поэтому p_eff = p + 1.
    X_train_aug = np.column_stack([
        np.ones(n),
        X_train
    ])

    X_query_aug = np.column_stack([
        np.ones(X_query.shape[0]),
        X_query
    ])

    xtx_inv = np.linalg.pinv(X_train_aug.T @ X_train_aug)

    leverage = np.sum(
        (X_query_aug @ xtx_inv) * X_query_aug,
        axis=1
    )

    p_eff = p + 1
    threshold = 3.0 * p_eff / n

    # Формально h не может быть больше 1 для обучающей OLS-H,
    # но при p >> n и псевдообратной матрице порог может быть > 1.
    # Для отображения оставляем классическую формулу.
    status = [
        "в AD" if h <= threshold else "вне AD"
        for h in leverage
    ]

    return {
        "leverage": leverage,
        "threshold": float(threshold),
        "p": int(p),
        "n": int(n),
        "status": status
    }
    
def qspr_make_ad_table(
    X_train,
    smiles,
    y=None,
    original_indices=None,
    desc_names=None
):
    """
    Таблица Applicability Domain для обучающей выборки.
    """
    ad = qspr_calculate_leverage_ad(
        X_train=X_train,
        X_query=None,
        desc_names=desc_names
    )

    n = len(ad["leverage"])

    table = pd.DataFrame({
        "№": range(1, n + 1),
        "Leverage h": ad["leverage"],
        "Порог h*": ad["threshold"],
        "AD-статус": ad["status"],
    })

    if original_indices is not None:
        table.insert(1, "Номер в исходной таблице", [int(i) + 1 for i in original_indices])

    if smiles is not None:
        insert_pos = 2 if original_indices is not None else 1
        table.insert(insert_pos, "SMILES", list(smiles))

    if y is not None:
        table["Значение свойства"] = y

    table["Надёжность по AD"] = table["AD-статус"].map({
        "в AD": "структурно в области модели",
        "вне AD": "экстраполяция: прогноз менее надёжен"
    })

    return table, ad

def qspr_count_outside_ad_for_model(model_data):
    """Считает число обучающих веществ вне leverage AD для сохранённой модели."""
    try:
        X_model = model_data.get("X_scaled", None)

        if X_model is None:
            return np.nan, np.nan

        ad = qspr_calculate_leverage_ad(X_train=X_model)
        leverage = np.asarray(ad["leverage"], dtype=float)
        threshold = float(ad["threshold"])
        n_out = int(np.sum(leverage > threshold))
        percent_out = n_out / len(leverage) * 100 if len(leverage) else np.nan
        return n_out, percent_out
    except Exception:
        return np.nan, np.nan    

def ad_williams_summary(
    williams_df
):
    return {
        "total":
            len(williams_df),

        "outside_ad":
            int(
                williams_df["Outside AD"].sum()
            ),

        "high_residual":
            int(
                williams_df["High Residual"].sum()
            ),

        "critical":
            int(
                williams_df["Critical Point"].sum()
            ),
    }