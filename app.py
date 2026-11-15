import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# ==================== 1. 全局页面配置与现代工业级 UI 样式 ====================
st.set_page_config(
    page_title="盾构推力预测轻量化智能体 V4.0 Ultra",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140

st.markdown("""
<style>
    .hero-container {
        background: linear-gradient(125deg, #0b132b 0%, #1c2541 48%, #3a506b 100%);
        color: #ffffff;
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(11, 19, 43, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 14.5px;
        color: #d8e2dc;
        margin: 0;
        line-height: 1.6;
    }
    .status-pill {
        display: inline-block;
        background: rgba(91, 192, 190, 0.22);
        border: 1px solid #5bc0be;
        color: #6fffe9;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12.5px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
    }
    .diag-card {
        background: #ffffff;
        border-left: 5px solid #1d3557;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
        border-top: 1px solid #f1f3f5;
        border-right: 1px solid #f1f3f5;
        border-bottom: 1px solid #f1f3f5;
    }
    .diag-card-title {
        font-weight: 700;
        color: #1d3557;
        font-size: 15.5px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .diag-score-badge {
        background: #e63946;
        color: white;
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .diag-card-body {
        color: #343a40;
        font-size: 13.8px;
        line-height: 1.65;
    }
    .order-chip {
        display: inline-block;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #0b132b;
        border: 1px solid #adb5bd;
        padding: 5px 14px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 13.5px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 2. 高性能缓存数据合成器 ====================
@st.cache_data(show_spinner=False)
def generate_synthetic_tbm_data(n_rings: int = 240, mutation_ratio: float = 0.25, noise_level: float = 0.04, seed: int = 2026) -> pd.DataFrame:
    """支持缓存的高真实感盾构掘进数据生成器（含渐变过渡带、地质突变、缺失值与物理越界毛刺）"""
    rng = np.random.default_rng(seed)
    rings = np.arange(1, n_rings + 1)

    n_m = int(n_rings * mutation_ratio)
    n_u = n_rings - n_m

    # 常规软土/粉土段与高刚度复合地层突变段（带3环平滑过渡）
    E_raw = np.concatenate([rng.uniform(5.2, 13.5, n_u), rng.uniform(24.0, 68.0, n_m)]) if n_m > 0 else rng.uniform(5.2, 13.5, n_rings)
    E = pd.Series(E_raw).rolling(window=3, min_periods=1).mean().values

    D = np.full(n_rings, rng.choice([6.28, 6.34, 6.40]))
    H = np.clip(np.linspace(rng.uniform(9.0, 10.5), rng.uniform(12.5, 16.5), n_rings) + rng.normal(0, 0.25, n_rings), 6.0, 26.0)
    v = rng.uniform(16.0, 52.0, n_rings)
    omega = np.concatenate([rng.uniform(0.6, 1.1, n_u), rng.uniform(1.0, 1.45, n_m)]) if n_m > 0 else rng.uniform(0.6, 1.1, n_rings)
    pe = rng.uniform(140.0, 305.0, n_rings)
    mu = np.concatenate([rng.uniform(0.29, 0.37, n_u), rng.uniform(0.21, 0.27, n_m)]) if n_m > 0 else rng.uniform(0.29, 0.37, n_rings)
    K0 = np.concatenate([rng.uniform(0.48, 0.58, n_u), rng.uniform(0.36, 0.45, n_m)]) if n_m > 0 else rng.uniform(0.48, 0.58, n_rings)
    phi = np.concatenate([rng.uniform(20.0, 27.0, n_u), rng.uniform(30.0, 39.0, n_m)]) if n_m > 0 else rng.uniform(20.0, 27.0, n_rings)
    gamma = np.concatenate([rng.uniform(18.2, 19.5, n_u), rng.uniform(20.2, 21.8, n_m)]) if n_m > 0 else rng.uniform(18.2, 19.5, n_rings)
    c = np.concatenate([rng.uniform(18.0, 30.0, n_u), rng.uniform(32.0, 50.0, n_m)]) if n_m > 0 else rng.uniform(18.0, 30.0, n_rings)
    W = rng.uniform(175.0, 340.0, n_rings)

    E_kPa = E * 1000.0
    v_m = v / 1000.0
    phi_rad = np.deg2rad(phi)

    true_theta = np.array([
        rng.uniform(0.75, 1.10),     # Π1: 开挖面主动支护
        rng.uniform(1.20, 1.95),     # Π2: 地层自稳能力
        rng.uniform(-1.55, -0.85),   # Π3: 初始地应力约束
        rng.uniform(-4.90, -2.90),   # Π4: 黏聚力剪切效应
        rng.uniform(0.018, 0.052),   # Π5: 内摩擦角剪切效应
        rng.uniform(0.006, 0.032),   # Π6: 土体压缩效应
        rng.uniform(1.85, 2.80)      # Π7: 岩土本构复合特性
    ])

    F_mech = (
        true_theta[0] * pe * (D ** 2)
        + true_theta[1] * W * (D ** 2)
        + true_theta[2] * (1.0 + K0) * gamma * H * (D ** 2)
        + true_theta[3] * (c * v_m / omega) * D
        + true_theta[4] * np.tan(phi_rad) * (v_m / omega) * E_kPa * D
        + true_theta[5] * (mu / ((1.0 + mu) * (1.0 - 2.0 * mu))) * (v_m / omega) * E_kPa * D
        + true_theta[6] * (2.0 * c * (1.0 + mu) / np.tan(phi_rad)) * (D ** 2)
    )
    F_base = np.abs(F_mech) + 7500.0 * (np.abs(F_mech) < 6000.0)

    # 叠加突变段未建模盾壳摩阻力、二阶非线性项与传感器噪声
    unmodeled_friction = np.concatenate([np.zeros(n_u), rng.uniform(950.0, 2300.0, n_m)]) if n_m > 0 else np.zeros(n_rings)
    nonlinear_coupling = 0.035 * (pe / E) ** 2 * (D ** 2)
    sensor_noise = rng.normal(0, noise_level * np.mean(F_base), n_rings)

    F = np.clip(F_base + unmodeled_friction + nonlinear_coupling + sensor_noise, 6000.0, 34000.0)

    df = pd.DataFrame({
        'Ring': rings, 'E': np.round(E, 3), 'D': D, 'omega': np.round(omega, 3),
        'H': np.round(H, 2), 'v': np.round(v, 2), 'mu': np.round(mu, 3),
        'K0': np.round(K0, 3), 'phi': np.round(phi, 2), 'pe': np.round(pe, 2),
        'gamma': np.round(gamma, 2), 'c': np.round(c, 2), 'W': np.round(W, 2),
        'F': np.round(F, 1)
    })

    # 注入3处缺失值与2处违背物理约束的异常值用于检验预处理能力
    if n_rings >= 40:
        na_idx = rng.choice(n_rings - 5, size=3, replace=False)
        df.loc[na_idx[0], 'pe'] = np.nan
        df.loc[na_idx[1], 'v'] = np.nan
        df.loc[na_idx[2], 'c'] = np.nan
        df.loc[int(n_rings * 0.18), 'mu'] = 0.62   # 泊松比越界 (>=0.5)
        df.loc[int(n_rings * 0.52), 'E'] = -3.2    # 弹性模量越界 (<=0)

    return df


# ==================== 3. 高性能缓存核心建模与诊断流水线 ====================
REQUIRED_COLS = ['E', 'D', 'omega', 'H', 'v', 'mu', 'K0', 'phi', 'pe', 'gamma', 'c', 'W', 'F']
PI_NAMES = ['Π1', 'Π2', 'Π3', 'Π4', 'Π5', 'Π6', 'Π7']
PI_METADATA = [
    ("Π1", "开挖面稳定控制", "pe / E", "开挖面土舱压力主动支护效应"),
    ("Π2", "开挖面稳定控制", "W / E", "周围地层三维应力自稳承载能力"),
    ("Π3", "环境地应力约束", "(1+K0)γH / E", "垂直与水平初始地应力场约束效应"),
    ("Π4", "刀盘-土体动态交互", "(c/E)·(v/ωD)", "黏聚力主导的掘进土体剪切效应"),
    ("Π5", "刀盘-土体动态交互", "tan(φ)·(v/ωD)", "内摩擦角主导的掘进土体剪切效应"),
    ("Π6", "刀盘-土体动态交互", "[μ/((1+μ)(1-2μ))]·(v/ωD)", "刀盘推进挤压引起的土体压缩效应"),
    ("Π7", "岩土本构复合特性", "2c(1+μ) / (E·tanφ)", "剪压复合破坏下的岩土本构特征")
]

@st.cache_data(show_spinner=False)
def run_cached_agent_pipeline(raw_df: pd.DataFrame, test_size: float, cv_folds: int, run_baselines: bool):
    """缓存核心计算结果，使得切换标签页和拖动决策滑块达到毫秒级响应"""
    t0 = time.perf_counter()

    # Step 1: 特征完整性检查
    missing = [c for c in REQUIRED_COLS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"数据表缺失以下核心物理特征字段：{missing}")

    df = raw_df.copy()
    if 'Ring' not in df.columns:
        df['Ring'] = np.arange(1, len(df) + 1)

    # Step 2: 缺失值统计与线性插值修复
    na_counts = df[REQUIRED_COLS].isna().sum().to_dict()
    total_na = int(sum(na_counts.values()))
    df[REQUIRED_COLS] = df[REQUIRED_COLS].interpolate(method='linear').bfill().ffill()

    # Step 3: 固体力学物理边界异常拦截
    valid_mask = (
        (df['E'] > 0) & (df['D'] > 0) & (df['omega'] > 0) &
        (df['mu'] > 0) & (df['mu'] < 0.5) &
        (df['phi'] > 0) & (df['phi'] < 90) & (df['F'] > 0)
    )
    outlier_rings = df.loc[~valid_mask, 'Ring'].tolist()
    df_clean = df[valid_mask].reset_index(drop=True)

    # Step 4: 国际标准单位制 (SI) 强制统一转换
    df_clean['E_kPa'] = df_clean['E'] * 1000.0          # MPa -> kPa
    df_clean['v_m_min'] = df_clean['v'] / 1000.0        # mm/min -> m/min
    df_clean['phi_rad'] = np.deg2rad(df_clean['phi'])   # deg -> rad

    clean_report = {
        "raw_rows": len(raw_df),
        "clean_rows": len(df_clean),
        "total_na_filled": total_na,
        "na_detail": {k: v for k, v in na_counts.items() if v > 0},
        "outlier_count": len(outlier_rings),
        "outlier_rings": outlier_rings
    }

    # Step 5: 构造基础无量纲 π 群与 7 个主导力学因子 Π1~Π7
    E, D, w = df_clean['E_kPa'], df_clean['D'], df_clean['omega']
    v, mu, phi = df_clean['v_m_min'], df_clean['mu'], df_clean['phi_rad']

    basic_pi = pd.DataFrame({
        'π1 (H/D)': df_clean['H'] / D,
        'π2 (v/ωD)': v / (w * D),
        'π3 (μ)': mu,
        'π4 (K0)': df_clean['K0'],
        'π5 (φ_rad)': phi,
        'π6 (pe/E)': df_clean['pe'] / E,
        'π7 (γD/E)': df_clean['gamma'] * D / E,
        'π8 (c/E)': df_clean['c'] / E,
        'π9 (W/E)': df_clean['W'] / E,
        'πF (F/ED²)': df_clean['F'] / (E * (D ** 2))
    })

    X_pi = pd.DataFrame(index=df_clean.index)
    X_pi['Π1'] = df_clean['pe'] / E
    X_pi['Π2'] = df_clean['W'] / E
    X_pi['Π3'] = (1.0 + df_clean['K0']) * df_clean['gamma'] * df_clean['H'] / E
    X_pi['Π4'] = (df_clean['c'] / E) * (v / (w * D))
    X_pi['Π5'] = np.tan(phi) * (v / (w * D))
    X_pi['Π6'] = (mu / ((1.0 + mu) * (1.0 - 2.0 * mu))) * (v / (w * D))
    X_pi['Π7'] = (2.0 * df_clean['c'] * (1.0 + mu)) / (E * np.tan(phi))
    y_pi = df_clean['F'] / (E * (D ** 2))

    # Step 6: 训练/测试集划分与无截距加性线性 LassoCV 辨识
    idx_train, idx_test = train_test_split(df_clean.index, test_size=test_size, random_state=42)
    X_train, X_test = X_pi.loc[idx_train], X_pi.loc[idx_test]
    y_train, y_test = y_pi.loc[idx_train], y_pi.loc[idx_test]

    # 等方差无偏竞争缩放 + 物理尺度逆还原
    col_std = X_train.std(axis=0).replace(0, 1.0).values
    X_train_norm = X_train.values / col_std
    X_test_norm = X_test.values / col_std

    alphas = np.logspace(-7, -1, 180)
    model = LassoCV(alphas=alphas, fit_intercept=False, cv=cv_folds, max_iter=50000, n_jobs=-1)
    model.fit(X_train_norm, y_train.values)

    best_lambda = float(model.alpha_)
    theta_norm = model.coef_
    theta_phys = model.coef_ / col_std

    # Step 7: Lasso 路径追踪
    alphas_path, coef_path_norm, _ = lasso_path(X_train_norm, y_train.values, alphas=alphas)
    pred_path = X_train_norm @ coef_path_norm
    mse_path = np.mean((pred_path - y_train.values[:, None]) ** 2, axis=0)
    n_feat_path = np.sum(np.abs(coef_path_norm) > 1e-7, axis=0)

    entry_step = [np.argmax(np.abs(coef_path_norm[i, :]) > 1e-7) for i in range(7)]
    entry_order = [PI_NAMES[i] for i in np.argsort(entry_step)]

    # Step 8: 还原推力预测值并计算多维评价指标 (R2, MAPE, RMSE, MAE)
    scale_tr = df_clean.loc[idx_train, 'E_kPa'].values * (df_clean.loc[idx_train, 'D'].values ** 2)
    scale_te = df_clean.loc[idx_test, 'E_kPa'].values * (df_clean.loc[idx_test, 'D'].values ** 2)

    F_train_true = df_clean.loc[idx_train, 'F'].values
    F_train_pred = (X_train.values @ theta_phys) * scale_tr
    F_test_true = df_clean.loc[idx_test, 'F'].values
    F_test_pred = (X_test.values @ theta_phys) * scale_te

    def calc_metrics(y_t, y_p):
        return {
            "R2": float(r2_score(y_t, y_p)),
            "MAPE": float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "MAE": float(mean_absolute_error(y_t, y_p))
        }

    metrics_train = calc_metrics(F_train_true, F_train_pred)
    metrics_test = calc_metrics(F_test_true, F_test_pred)

    # 可选：运行黑盒对比基线 (SVR & MLP)
    baselines = None
    if run_baselines:
        feat_12 = ['E', 'D', 'omega', 'H', 'v', 'mu', 'K0', 'phi', 'pe', 'gamma', 'c', 'W']
        raw_scaler = StandardScaler()
        X_raw_tr = raw_scaler.fit_transform(df_clean.loc[idx_train, feat_12])
        X_raw_te = raw_scaler.transform(df_clean.loc[idx_test, feat_12])

        svr = SVR(kernel='rbf', C=100.0)
        svr.fit(X_raw_tr, F_train_true)
        svr_pred = svr.predict(X_raw_te)

        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)
        mlp.fit(X_raw_tr, F_train_true)
        mlp_pred = mlp.predict(X_raw_te)

        baselines = {
            "SVR": calc_metrics(F_test_true, svr_pred),
            "MLP": calc_metrics(F_test_true, mlp_pred)
        }

    # Step 9: 六维偏差定量归因分析
    err = F_test_pred - F_test_true
    rel_err = err / F_test_true
    abs_rel_err = np.abs(rel_err)

    anomaly_rate = (clean_report['total_na_filled'] + clean_report['outlier_count']) / max(1, clean_report['raw_rows']) * 100.0
    v_cv = float(df_clean['v'].std() / df_clean['v'].mean() * 100.0)

    pca3 = PCA(n_components=3)
    Z_tr = pca3.fit_transform(X_train_norm)
    Z_te = pca3.transform(X_test_norm)
    center = Z_tr.mean(axis=0, keepdims=True)
    r95 = float(np.percentile(cdist(Z_tr, center), 95))
    dist_te = cdist(Z_te, center).ravel()
    extrap_mask = dist_te > r95
    extrap_rate = float(np.mean(extrap_mask) * 100.0)

    n_train = len(idx_train)
    active_factors = int(np.sum(np.abs(theta_norm) > 1e-6))

    quad_r, _ = pearsonr(F_test_pred ** 2, -err)
    over_rate = float(np.mean(rel_err > 0.06) * 100.0)
    under_rate = float(np.mean(rel_err < -0.06) * 100.0)

    E_grad_all = np.abs(df_clean['E'].diff() / df_clean['E']).fillna(0)
    E_grad_te = E_grad_all.loc[idx_test].values
    jump_r, _ = pearsonr(E_grad_te, abs_rel_err)

    reasons = []
    for i in range(len(rel_err)):
        tags = []
        if abs_rel_err[i] <= 0.05:
            reasons.append("✅ 正常范围内 (误差≤5%)")
            continue
        if extrap_mask[i]:
            tags.append("工况外推越界")
        if E_grad_te[i] > 0.18:
            tags.append("地层模量突变")
        if rel_err[i] < -0.06:
            tags.append("预测偏小(未覆盖盾壳摩阻/结泥饼)")
        elif rel_err[i] > 0.06:
            tags.append("预测偏大(线性截断/勘察阶跃)")
        reasons.append(" | ".join(tags) if tags else "传感器高频随机波动")

    detail_df = df_clean.loc[idx_test, ['Ring', 'E', 'H', 'v', 'omega', 'pe', 'F']].copy()
    detail_df['预测推力 F_pred (kN)'] = np.round(F_test_pred, 1)
    detail_df['绝对偏差 (kN)'] = np.round(err, 1)
    detail_df['相对误差幅值 (%)'] = np.round(abs_rel_err * 100.0, 2)
    detail_df['相对偏差 (%)'] = np.round(rel_err * 100.0, 2)
    detail_df['偏差主导原因诊断'] = reasons
    detail_df = detail_df.sort_values('Ring').reset_index(drop=True)

    radar_scores = {
        "① 数据质量与匹配噪声": round(min(100.0, anomaly_rate * 7.5 + v_cv * 1.3), 1),
        "② 工况外推程度": round(min(100.0, extrap_rate * 4.2 + 18.0), 1),
        "③ 样本规模约束": round(min(100.0, max(15.0, (320.0 - n_train) * 0.35)), 1),
        "④ 线性假设截断": round(min(100.0, abs(quad_r) * 165.0 + 22.0), 1),
        "⑤ 因子机制未覆盖度": round(min(100.0, under_rate * 2.4 + 20.0), 1),
        "⑥ 地质刚度突变": round(min(100.0, abs(jump_r) * 150.0 + np.mean(E_grad_te > 0.15) * 110.0), 1)
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "elapsed_ms": elapsed_ms,
        "clean_report": clean_report,
        "df_clean": df_clean,
        "basic_pi": basic_pi,
        "X_pi": X_pi,
        "y_pi": y_pi,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "best_lambda": best_lambda,
        "theta_norm": theta_norm,
        "theta_phys": theta_phys,
        "alphas_path": alphas_path,
        "coef_path_norm": coef_path_norm,
        "mse_path": mse_path,
        "n_feat_path": n_feat_path,
        "entry_order": entry_order,
        "F_train_true": F_train_true,
        "F_train_pred": F_train_pred,
        "F_test_true": F_test_true,
        "F_test_pred": F_test_pred,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "baselines": baselines,
        "diag": {
            "anomaly_rate": float(anomaly_rate),
            "v_cv": v_cv,
            "extrap_rate": extrap_rate,
            "n_train": n_train,
            "active_factors": active_factors,
            "quad_r": float(abs(quad_r)),
            "over_rate": over_rate,
            "under_rate": under_rate,
            "jump_r": float(abs(jump_r)),
            "Z_tr": Z_tr,
            "Z_te": Z_te,
            "extrap_mask": extrap_mask,
            "detail_df": detail_df,
            "radar_scores": radar_scores
        }
    }


# ==================== 4. 侧边栏控制中枢与顶部状态横幅 ====================
if "rand_seed" not in st.session_state:
    st.session_state.rand_seed = 2026

with st.sidebar:
    st.header("🎛️ 智能体控制台 (V4.0 Ultra)")
    source_mode = st.radio(
        "1. 选择工程数据接入方式：",
        ["🎲 随机生成一组盾构工程数据", "📂 上传本地真实工程数据表"]
    )

    if source_mode == "🎲 随机生成一组盾构工程数据":
        n_rings_input = st.slider("生成掘进总环数 (样本规模)", 80, 600, 240, 20)
        mutation_input = st.slider("地质突变区段占比", 0.0, 0.50, 0.25, 0.05)
        noise_input = st.slider("传感器测量噪声水平", 0.01, 0.10, 0.04, 0.01)

        if st.button("🔄 一键随机生成全新工况数据", use_container_width=True, type="primary"):
            st.session_state.rand_seed = int(np.random.randint(1000, 99999))

        raw_df = generate_synthetic_tbm_data(
            n_rings=n_rings_input,
            mutation_ratio=mutation_input,
            noise_level=noise_input,
            seed=st.session_state.rand_seed
        )
        st.caption(f"当前随机种子：`{st.session_state.rand_seed}`（已启用 `@st.cache_data` 内存加速）")
        st.download_button(
            "⬇️ 导出当前随机工程数据 (.csv)",
            data=raw_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"tbm_random_seed{st.session_state.rand_seed}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        uploaded = st.file_uploader("上传包含 13 个核心字段的 CSV 或 XLSX", type=["csv", "xlsx"])
        template_df = generate_synthetic_tbm_data(n_rings=20, seed=100)
        st.download_button(
            "📄 下载标准表头数据模板 (.csv)",
            data=template_df.to_csv(index=False).encode('utf-8-sig'),
            file_name="tbm_standard_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        else:
            st.info("👈 请上传您的 Excel/CSV 数据文件，或切换至【随机生成一组盾构工程数据】。")
            st.stop()

    st.divider()
    st.subheader("2. 算法与基线对比配置")
    test_ratio = st.slider("测试集划分比例 (Test Split)", 0.15, 0.40, 0.25, 0.05)
    cv_folds = st.selectbox("Lasso 交叉验证折数 (CV)", [3, 5, 10], index=1)
    run_baselines = st.checkbox("同台对比 SVR / MLP 纯黑盒模型", value=True)

# 执行缓存流水线
try:
    res = run_cached_agent_pipeline(raw_df, test_ratio, cv_folds, run_baselines)
except Exception as e:
    st.error(f"❌ 数据检查未通过：{e}")
    st.stop()

cr = res["clean_report"]
mt_tr = res["metrics_train"]
mt_te = res["metrics_test"]
diag = res["diag"]
t_phys = res["theta_phys"]

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🛡️ 力学逻辑引导的盾构推力预测轻量化智能体 (V4.0 Ultra)</div>
    <div class="hero-subtitle">
        全流程自动化数据治理 · 白金汉 π 无量纲参量构造 · 无截距加性线性 LassoCV 参数辨识 · 六维偏差定量雷达归因 · 毫秒级施工推力预演
    </div>
    <div>
        <span class="status-pill">⚡ 引擎计算耗时：{res['elapsed_ms']:.1f} ms (缓存加速已开启)</span>
        <span class="status-pill">🧹 样本净化：{cr['raw_rows']} 环 ➔ {cr['clean_rows']} 环</span>
        <span class="status-pill">🏆 主导激活机制：{res['entry_order'][0]} ➔ {res['entry_order'][1]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 顶部全局 KPI 仪表盘
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("有效建模样本", f"{cr['clean_rows']} 环", f"插补{cr['total_na_filled']}处 / 拦截异常{cr['outlier_count']}环")
k2.metric("测试集决定系数 R²", f"{mt_te['R2']:.4f}", f"训练集 R²: {mt_tr['R2']:.4f}")
k3.metric("测试集平均误差 MAPE", f"{mt_te['MAPE']:.2f}%", f"训练集 MAPE: {mt_tr['MAPE']:.2f}%")
k4.metric("测试集均方根误差 RMSE", f"{mt_te['RMSE']:.1f} kN", f"MAE: {mt_te['MAE']:.1f} kN")
k5.metric("最优正则化超参数 λ*", f"{res['best_lambda']:.2e}", f"激活 {diag['active_factors']}/7 个力学因子")

# ==================== 5. 五大核心工作区标签页 ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 数据治理与单位统一",
    "2️⃣ 量纲矩阵与无量纲参量",
    "3️⃣ 参数识别、Lasso路径与黑盒对比",
    "4️⃣ 实测对比与六维偏差雷达归因",
    "5️⃣ 毫秒级施工决策预演与报告导出"
])

# -------------------- Tab 1: 数据预处理 --------------------
with tab1:
    st.subheader("模块 1：特征完整性检查、缺失/异常值修复与国际标准单位制 (SI) 统一")
    c_left, c_right = st.columns([1.15, 1.85])
    with c_left:
        st.markdown("##### （1）自动化数据质量体检日志")
        st.success("✅ **特征完整性校验通过**：已集齐全部 12 项输入特征与目标推力 $F$。")
        st.info(
            f"📊 **预处理执行明细**：\n"
            f"- 原始输入总环数：**{cr['raw_rows']} 环**\n"
            f"- 缺失值线性插值修复：**{cr['total_na_filled']} 处**（明细：{cr['na_detail'] or '无'}）\n"
            f"- 物理越界异常环剔除：**{cr['outlier_count']} 环**（剔除环号：{cr['outlier_rings'] or '无'}，拦截准则：泊松比 $\\mu \\in (0, 0.5)$、模量 $E > 0$、转速 $\\omega > 0$）\n"
            f"- 最终合格建模样本：**{cr['clean_rows']} 环**"
        )
        st.markdown("##### （2）国际标准单位制 (SI) 强制统一映射表")
        unit_table = pd.DataFrame({
            "变量符号": ["E", "pe, c, W", "D, H", "v", "omega", "phi", "gamma", "F"],
            "原始单位": ["MPa", "kPa", "m", "mm/min", "rev/min", "度 (°)", "kN/m³", "kN"],
            "统一后标准单位": ["kPa (×1000)", "kPa", "m", "m/min (÷1000)", "rev/min", "弧度 (rad)", "kN/m³", "kN"],
            "基本量纲 [M,L,T]": ["M¹L⁻¹T⁻²", "M¹L⁻¹T⁻²", "L¹", "L¹T⁻¹", "T⁻¹", "无量纲", "M¹L⁻²T⁻²", "M¹L¹T⁻²"]
        })
        st.dataframe(unit_table, use_container_width=True, hide_index=True)

    with c_right:
        st.markdown("##### （3）清洗后 13 项工程特征统计分布表")
        desc_df = res["df_clean"][REQUIRED_COLS].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
        st.dataframe(desc_df.style.format("{:.2f}"), use_container_width=True)
        st.markdown("##### （4）清洗并统一单位后的数据集前 10 环预览")
        st.dataframe(res["df_clean"][['Ring'] + REQUIRED_COLS].head(10).style.format("{:.2f}"), use_container_width=True, hide_index=True)

# -------------------- Tab 2: 无量纲参量构造 --------------------
with tab2:
    st.subheader("模块 2：基于白金汉 π 定理与四大力学机制的无量纲参量自动构造")
    st.markdown("选取岩土弹性模量 $E$、刀盘直径 $D$、刀盘转速 $\\omega$ 作为 3 个独立基本物理量，消去质量 $M$、长度 $L$、时间 $T$ 量纲，导出 **7 个表征主导力学逻辑的无量纲自变量 $\\Pi_1 \\sim \\Pi_7$** 与 **无量纲总推力 $\\Pi_F = F / (ED^2)$**：")

    info_df = pd.DataFrame(PI_METADATA, columns=["因子符号", "所属力学控制类别", "无量纲数学定义", "工程物理含义"])
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    col_p1, col_p2 = st.columns([1.45, 1.0])
    with col_p1:
        st.markdown("##### 自动构造的 7 个无量纲力学参量 ($\\Pi_1 \\sim \\Pi_7$) 与无量纲推力 ($\\Pi_F$) 数值表")
        st.dataframe(res["X_pi"].assign(Π_F=res["y_pi"]).head(12).style.format("{:.5e}"), use_container_width=True)
    with col_p2:
        st.markdown("##### 无量纲力学参量皮尔逊相关性热力图")
        fig_corr, ax_corr = plt.subplots(figsize=(5.8, 4.3), constrained_layout=True)
        corr_mat = res["X_pi"].assign(Π_F=res["y_pi"]).corr().values
        im_c = ax_corr.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
        labels_c = PI_NAMES + ['Π_F']
        ax_corr.set_xticks(range(8)); ax_corr.set_xticklabels(labels_c)
        ax_corr.set_yticks(range(8)); ax_corr.set_yticklabels(labels_c)
        for i in range(8):
            for j in range(8):
                ax_corr.text(j, i, f"{corr_mat[i, j]:.2f}", ha='center', va='center', fontsize=8, color='black')
        plt.colorbar(im_c, ax=ax_corr, fraction=0.046, pad=0.04)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

# -------------------- Tab 3: 参数识别与 Lasso 路径 --------------------
with tab3:
    st.subheader("模块 3：加性线性力学框架参数识别、Lasso 路径分析与黑盒模型对比")

    st.markdown("##### （1）数据驱动识别输出的显式盾构推力预测物理公式")
    st.latex(
        rf"F = {t_phys[0]:.3f}\,p_e D^2 + ({t_phys[1]:.3f})\,W D^2 + ({t_phys[2]:.3f})\,(1+K_0)\gamma H D^2 "
        rf"+ ({t_phys[3]:.3f})\,\frac{{cv}}{{\omega}}D + ({t_phys[4]:.4f})\,\tan\varphi\frac{{v}}{{\omega}}ED "
        rf"+ ({t_phys[5]:.4f})\,\frac{{\mu}}{{(1+\mu)(1-2\mu)}}\frac{{v}}{{\omega}}ED "
        rf"+ ({t_phys[6]:.3f})\,\frac{{2c(1+\mu)}}{{\tan\varphi}}D^2"
    )
    plain_eq = (
        f"F = {t_phys[0]:.4f}*pe*D^2 + ({t_phys[1]:.4f})*W*D^2 + ({t_phys[2]:.4f})*(1+K0)*gamma*H*D^2 "
        f"+ ({t_phys[3]:.4f})*(c*v/omega)*D + ({t_phys[4]:.5f})*tan(phi)*(v/omega)*E*D "
        f"+ ({t_phys[5]:.5f})*[mu/((1+mu)(1-2mu))]*(v/omega)*E*D + ({t_phys[6]:.4f})*[2c(1+mu)/tan(phi)]*D^2"
    )
    st.code(plain_eq, language="text")

    st.markdown("##### （2）Lasso 正则化路径特征激活顺序（反映当前地质工况下的主导力学机制）")
    chips_html = "".join([f'<span class="order-chip">第 {i+1} 顺位：{name}</span>' for i, name in enumerate(res["entry_order"])])
    st.markdown(chips_html, unsafe_allow_html=True)

    fig_lasso, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(16, 4.8), constrained_layout=True)
    palette = ['#0077b6', '#e63946', '#2a9d8f', '#f4a261', '#8338ec', '#fb8500', '#3a86ff']
    for i in range(7):
        ax_l1.plot(res["alphas_path"], res["coef_path_norm"][i, :], color=palette[i], linewidth=2.1, label=PI_NAMES[i])
    ax_l1.axvline(res["best_lambda"], color='#d90429', linestyle='--', linewidth=1.8, label=f"最优 $\\lambda^*$ = {res['best_lambda']:.2e}")
    ax_l1.set_xscale('log'); ax_l1.invert_xaxis()
    ax_l1.set_xlabel("正则化超参数 $\\lambda$ (由强惩罚到弱惩罚)")
    ax_l1.set_ylabel("标准化无量纲因子权重系数")
    ax_l1.set_title("(a) Lasso 正则化路径：无量纲因子权重演化轨迹")
    ax_l1.legend(ncol=2, fontsize=9)
    ax_l1.grid(True, linestyle='--', alpha=0.4)

    ax_l2_twin = ax_l2.twinx()
    ax_l2.plot(res["alphas_path"], res["mse_path"], 'b-', linewidth=2.1, label='均方误差 (MSE)')
    ax_l2.axvline(res["best_lambda"], color='#d90429', linestyle='--', linewidth=1.8, label=f"最优 $\\lambda^*$ = {res['best_lambda']:.2e}")
    ax_l2_twin.step(res["alphas_path"], res["n_feat_path"], where='post', color='#2b9348', linewidth=2.1, label='激活特征数')
    ax_l2.set_xscale('log'); ax_l2.invert_xaxis()
    ax_l2.set_xlabel("正则化超参数 $\\lambda$ (由强惩罚到弱惩罚)")
    ax_l2.set_ylabel("无量纲均方误差 MSE", color='b')
    ax_l2_twin.set_ylabel("激活的无量纲力学因子数量", color='#2b9348')
    ax_l2_twin.set_yticks(range(0, 8))
    ax_l2.set_title("(b) 正则化超参数寻优曲线与入选特征阶跃图")
    ax_l2.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig_lasso)
    plt.close(fig_lasso)

    if res["baselines"] is not None:
        st.markdown("##### （3）同台竞技：本文力学逻辑引导模型 vs 纯数据驱动黑盒基线模型（SVR / MLP）")
        b_svr = res["baselines"]["SVR"]
        b_mlp = res["baselines"]["MLP"]
        arena_df = pd.DataFrame({
            "对比模型": [
                "Proposed Method (本文力学逻辑引导加性线性模型)",
                "SVR (支持向量回归 - 12维原始特征黑盒)",
                "MLP (双隐层神经网络 - 12维原始特征黑盒)"
            ],
            "测试集 R²": [mt_te["R2"], b_svr["R2"], b_mlp["R2"]],
            "测试集 MAPE (%)": [mt_te["MAPE"], b_svr["MAPE"], b_mlp["MAPE"]],
            "测试集 RMSE (kN)": [mt_te["RMSE"], b_svr["RMSE"], b_mlp["RMSE"]],
            "测试集 MAE (kN)": [mt_te["MAE"], b_svr["MAE"], b_mlp["MAE"]],
            "物理可解释性与工程迁移特性": [
                "✅ 具备显式物理方程与因果权重，跨地质抗过拟合能力强",
                "❌ 核函数隐式映射无物理公式，地层突变时易偏移",
                "❌ 权重矩阵完全黑盒，小样本及外推场景下极易过拟合"
            ]
        })
        st.dataframe(
            arena_df.style.format({"测试集 R²": "{:.4f}", "测试集 MAPE (%)": "{:.2f}%", "测试集 RMSE (kN)": "{:.1f}", "测试集 MAE (kN)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True
        )

# -------------------- Tab 4: 实测对比与六维偏差归因 --------------------
with tab4:
    st.subheader("模块 4：实测推力对比分析与六大核心因素偏差定量归因")

    # 彻底修复坐标轴间距与 R^2 上标渲染的三联科研图
    fig_comp = plt.figure(figsize=(18, 5.2), constrained_layout=True)
    gs_c = gridspec.GridSpec(1, 3, figure=fig_comp, width_ratios=[1.55, 1.0, 1.1])

    # 子图1：逐环推力对比与误差柱状图
    ax_c1 = fig_comp.add_subplot(gs_c[0])
    order_te = np.argsort(res["df_clean"].loc[res["idx_test"], 'Ring'].values)
    rings_te = res["df_clean"].loc[res["idx_test"], 'Ring'].values[order_te]
    f_true_te = res["F_test_true"][order_te]
    f_pred_te = res["F_test_pred"][order_te]
    abs_err_te = np.abs(f_true_te - f_pred_te)

    ax_c1_err = ax_c1.twinx()
    ax_c1_err.bar(rings_te, abs_err_te, width=max(1.5, len(rings_te)*0.03), color='#90e0ef', alpha=0.45, label='绝对偏差 |ΔF|')
    ax_c1_err.set_ylabel("绝对误差 Error (kN)", color='#0077b6')
    ax_c1_err.set_ylim(0, max(abs_err_te) * 2.6)

    ax_c1.plot(rings_te, f_true_te, 'o-', color='#d62828', markersize=4, linewidth=1.7, label='实测推力 True')
    ax_c1.plot(rings_te, f_pred_te, 's-', color='#003049', markersize=4, linewidth=1.7, label='预测推力 Pred')
    ax_c1.set_xlabel("掘进环号 (Ring Number)")
    ax_c1.set_ylabel("盾构推力 Thrust (kN)")
    ax_c1.set_title("(a) 测试集实测推力与预测推力逐环对比")
    ax_c1.legend(loc='upper left', fontsize=8.5)
    ax_c1_err.legend(loc='upper right', fontsize=8.5)
    ax_c1.grid(True, linestyle='--', alpha=0.4)

    # 子图2：1:1 对角散点图（使用 LaTeX $R^2$ 彻底杜绝方框乱码）
    ax_c2 = fig_comp.add_subplot(gs_c[1])
    low_f, high_f = min(f_true_te.min(), f_pred_te.min()) * 0.92, max(f_true_te.max(), f_pred_te.max()) * 1.08
    ax_c2.plot([low_f, high_f], [low_f, high_f], 'k--', linewidth=1.4, label='1:1 理想线')
    ax_c2.fill_between([low_f, high_f], [low_f*0.92, high_f*0.92], [low_f*1.08, high_f*1.08], color='gray', alpha=0.15, label='±8% 误差带')
    ax_c2.scatter(f_true_te, f_pred_te, c='#2a9d8f', edgecolors='k', s=45, alpha=0.85, label='测试集样本')
    ax_c2.set_xlim(low_f, high_f); ax_c2.set_ylim(low_f, high_f)
    ax_c2.set_xlabel("实测推力 True Thrust (kN)")
    ax_c2.set_ylabel("预测推力 Predicted Thrust (kN)")
    ax_c2.set_title(f"(b) 实测 vs 预测散点分布 ($R^2$ = {mt_te['R2']:.3f})")
    ax_c2.legend(loc='upper left', fontsize=8.5)
    ax_c2.grid(True, linestyle='--', alpha=0.4)

    # 子图3：3D PCA 物理特征空间工况外推检测
    ax_c3 = fig_comp.add_subplot(gs_c[2], projection='3d')
    Z_tr, Z_te, ext_mask = diag['Z_tr'], diag['Z_te'], diag['extrap_mask']
    ax_c3.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], c='#457b9d', s=18, alpha=0.45, label='训练集簇')
    ax_c3.scatter(Z_te[~ext_mask, 0], Z_te[~ext_mask, 1], Z_te[~ext_mask, 2], c='#2a9d8f', s=32, label='正常测试环')
    if np.any(ext_mask):
        ax_c3.scatter(Z_te[ext_mask, 0], Z_te[ext_mask, 1], Z_te[ext_mask, 2], c='#e63946', marker='^', s=55, label='外推越界环')
    ax_c3.set_xlabel("PC1"); ax_c3.set_ylabel("PC2"); ax_c3.set_zlabel("PC3")
    ax_c3.set_title("(c) 3D Π因子特征空间与工况外推检测")
    ax_c3.legend(loc='upper left', fontsize=8)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.markdown("##### （2）六维偏差定量归因雷达图与机理性深度诊断")
    r_col1, r_col2 = st.columns([1.0, 1.85])

    with r_col1:
        # 绘制六维偏差影响度雷达图
        labels_r = list(diag["radar_scores"].keys())
        values_r = list(diag["radar_scores"].values())
        angles = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=False).tolist()
        values_plot = values_r + values_r[:1]
        angles_plot = angles + angles[:1]

        fig_rad, ax_rad = plt.subplots(figsize=(5.2, 5.0), subplot_kw=dict(polar=True), constrained_layout=True)
        ax_rad.plot(angles_plot, values_plot, 'o-', linewidth=2.2, color='#d62828')
        ax_rad.fill(angles_plot, values_plot, alpha=0.25, color='#e63946')
        ax_rad.set_thetagrids(np.degrees(angles), labels_r, fontsize=9.5)
        ax_rad.set_ylim(0, 100)
        ax_rad.set_title("六大偏差来源敏感度定量评分 (0-100)", pad=18, fontsize=11, fontweight='bold')
        st.pyplot(fig_rad)
        plt.close(fig_rad)

    with r_col2:
        sc = diag["radar_scores"]
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown(f"""
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>① 数据质量因素</span>
                    <span class="diag-score-badge">影响度 {sc['① 数据质量与匹配噪声']}</span>
                </div>
                <div class="diag-card-body">
                    缺失/异常修复占比 <b>{diag['anomaly_rate']:.1f}%</b>，推进速度变异度 <b>{diag['v_cv']:.1f}%</b>。地质勘察为钻孔离散采样，而盾构传感器为连续高频采样，按环对齐时的“阶跃失真”引发了局部随机偏大或偏小。
                </div>
            </div>
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>② 工况外推程度</span>
                    <span class="diag-score-badge">影响度 {sc['② 工况外推程度']}</span>
                </div>
                <div class="diag-card-body">
                    在 3D $\\Pi$ 因子物理特征空间中，测试集有 <b>{diag['extrap_rate']:.1f}%</b> 的样本超出训练集 95% 置信包络半径（红三角）。处于外推边界的样本缺乏同分布训练点支撑，易产生系统性偏离。
                </div>
            </div>
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>③ 样本规模因素</span>
                    <span class="diag-score-badge">影响度 {sc['③ 样本规模约束']}</span>
                </div>
                <div class="diag-card-body">
                    当前训练集为 <b>{diag['n_train']} 环</b>，激活 <b>{diag['active_factors']}/7</b> 个因子。小样本下 L1 正则化（$\\lambda^*={res['best_lambda']:.1e}$）的权重收缩效应会削平极端高推力峰值（导致峰值<b>预测偏小</b>）并抬高极低推力谷值（导致谷值<b>预测偏大</b>）。
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sub_c2:
            st.markdown(f"""
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>④ 线性假设因素</span>
                    <span class="diag-score-badge">影响度 {sc['④ 线性假设截断']}</span>
                </div>
                <div class="diag-card-body">
                    二阶非线性残差相关度 <b>r = {diag['quad_r']:.2f}</b>。加性线性框架 $\\Pi_F^* = \\sum \\theta_i^* \\Pi_i$ 保障了物理可解释性与迁移性，但在软硬岩土交界处忽略了高阶非线性交叉耦合，产生截断误差。
                </div>
            </div>
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>⑤ 无量纲因子机制覆盖</span>
                    <span class="diag-score-badge">影响度 {sc['⑤ 因子机制未覆盖度']}</span>
                </div>
                <div class="diag-card-body">
                    测试集预测偏小占比 <b>{diag['under_rate']:.1f}%</b>，偏大占比 <b>{diag['over_rate']:.1f}%</b>。现有 7 个因子聚焦正面支护与切削压缩；当出现连续多环<b>预测明显偏小（实测推力偏大）</b>时，指示存在盾壳摩阻力剧增、结泥饼或刀具磨损等未覆盖机制。
                </div>
            </div>
            <div class="diag-card">
                <div class="diag-card-title">
                    <span>⑥ 地质突变因素</span>
                    <span class="diag-score-badge">影响度 {sc['⑥ 地质刚度突变']}</span>
                </div>
                <div class="diag-card-body">
                    相邻环弹性模量突变率 $|\\Delta E/E|$ 与误差相关系数达 <b>r = {diag['jump_r']:.2f}</b>。当盾构突入高刚度复合地层时，围岩模量 $E$ 成倍跃升引发瞬态推力冲击，造成峰值响应偏差。
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("##### （3）测试集逐环实测对比与偏差原因自动定位明细表（带误差进度条可视化）")
    st.dataframe(
        diag["detail_df"],
        column_config={
            "相对误差幅值 (%)": st.column_config.ProgressColumn(
                "相对误差幅值 (%)",
                help="测试环预测推力与实测推力的相对误差绝对值",
                format="%.2f%%",
                min_value=0.0,
                max_value=float(max(20.0, diag["detail_df"]["相对误差幅值 (%)"].max()))
            )
        },
        use_container_width=True,
        hide_index=True
    )

# -------------------- Tab 5: 施工辅助决策与报告导出 --------------------
with tab5:
    st.subheader("模块 5：毫秒级施工辅助决策推力预演台 & 课题成果报告一键导出")
    st.markdown("得益于 `@st.cache_data` 内存缓存架构，拖动下方任意地质或掘进滑块时**无需重新训练模型**，系统将在毫秒级内解算总推力及 7 个无量纲力学机制的分项贡献：")

    med = res["df_clean"].median()
    s1, s2, s3, s4 = st.columns(4)
    sim_E = s1.slider("地层弹性模量 E (MPa)", float(res["df_clean"]['E'].min()), float(res["df_clean"]['E'].max()), float(med['E']))
    sim_H = s2.slider("隧道埋深 H (m)", float(res["df_clean"]['H'].min()), float(res["df_clean"]['H'].max()), float(med['H']))
    sim_pe = s3.slider("土舱支护压力 pe (kPa)", float(res["df_clean"]['pe'].min()), float(res["df_clean"]['pe'].max()), float(med['pe']))
    sim_v = s4.slider("推进速度 v (mm/min)", float(res["df_clean"]['v'].min()), float(res["df_clean"]['v'].max()), float(med['v']))

    s5, s6, s7, s8 = st.columns(4)
    sim_w = s5.slider("刀盘转速 ω (rev/min)", float(res["df_clean"]['omega'].min()), float(res["df_clean"]['omega'].max()), float(med['omega']))
    sim_c = s6.slider("土体黏聚力 c (kPa)", float(res["df_clean"]['c'].min()), float(res["df_clean"]['c'].max()), float(med['c']))
    sim_phi = s7.slider("内摩擦角 φ (°)", float(res["df_clean"]['phi'].min()), float(res["df_clean"]['phi'].max()), float(med['phi']))
    sim_W = s8.slider("地基承载力 W (kPa)", float(res["df_clean"]['W'].min()), float(res["df_clean"]['W'].max()), float(med['W']))

    E_k, D_m, v_m, phi_r, mu_v, K0_v, gam_v = sim_E * 1000.0, med['D'], sim_v / 1000.0, np.deg2rad(sim_phi), med['mu'], med['K0'], med['gamma']
    sim_pi_vec = np.array([
        sim_pe / E_k,
        sim_W / E_k,
        (1.0 + K0_v) * gam_v * sim_H / E_k,
        (sim_c / E_k) * (v_m / (sim_w * D_m)),
        np.tan(phi_r) * (v_m / (sim_w * D_m)),
        (mu_v / ((1.0 + mu_v) * (1.0 - 2.0 * mu_v))) * (v_m / (sim_w * D_m)),
        (2.0 * sim_c * (1.0 + mu_v)) / (E_k * np.tan(phi_r))
    ])
    contrib_kN = sim_pi_vec * t_phys * E_k * (D_m ** 2)
    pred_total_kN = float(np.sum(contrib_kN))

    st.success(f"🚀 **实时预演总推力预测结果**：**{pred_total_kN:,.1f} kN** （刀盘直径 $D = {D_m:.2f}\\text{{ m}}$）")

    sim_col1, sim_col2 = st.columns([1.2, 1.3])
    with sim_col1:
        contrib_df = pd.DataFrame({
            "无量纲因子": PI_NAMES,
            "力学机制名称": [item[3] for item in PI_METADATA],
            "无量纲数值": sim_pi_vec,
            "分项推力贡献 (kN)": np.round(contrib_kN, 1)
        })
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    with sim_col2:
        fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
        bar_colors = ['#2a9d8f' if val >= 0 else '#e63946' for val in contrib_kN]
        ax_bar.barh(PI_NAMES, contrib_kN, color=bar_colors, edgecolor='k', height=0.6)
        ax_bar.axvline(0, color='black', linewidth=1.0)
        ax_bar.set_xlabel("分项推力贡献值 (kN)")
        ax_bar.set_title("各无量纲力学因子对当前总推力的分项贡献拆解")
        ax_bar.grid(True, axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    st.divider()
    st.markdown("##### 📥 一键导出完整课题诊断报告与测试集对比台账")
    report_md = f"""# 力学逻辑引导的盾构推力预测智能体 — 自动建模与诊断报告

## 一、 数据治理与预处理汇总
- **原始掘进环样本数**：{cr['raw_rows']} 环
- **缺失值线性插补修复**：{cr['total_na_filled']} 处
- **物理越界异常拦截**：{cr['outlier_count']} 环（拦截环号：{cr['outlier_rings']}）
- **最终有效建模样本数**：{cr['clean_rows']} 环

## 二、 显式推力预测物理表达式与模型评价指标
- **显式推力预测方程**：
  `{plain_eq}`
- **最优正则化超参数 (λ*)**：`{res['best_lambda']:.4e}`
- **Lasso 路径特征激活顺序**：`{' -> '.join(res['entry_order'])}`
- **测试集评价指标**：
  - 决定系数 $R^2$：`{mt_te['R2']:.4f}`
  - 平均绝对百分比误差 MAPE：`{mt_te['MAPE']:.2f}%`
  - 均方根误差 RMSE：`{mt_te['RMSE']:.2f} kN`
  - 平均绝对误差 MAE：`{mt_te['MAE']:.2f} kN`

## 三、 预测推力偏差偏大/偏小原因：六维定量归因分析
1. **数据质量因素（影响度评分 {diag['radar_scores']['① 数据质量与匹配噪声']}）**：缺失与异常修复占比 {diag['anomaly_rate']:.1f}%，推进速度变异度 {diag['v_cv']:.1f}%，勘察钻孔离散数据与连续传感器按环对齐存在阶跃失真。
2. **工况外推程度（影响度评分 {diag['radar_scores']['② 工况外推程度']}）**：3D PCA 物理特征空间中有 {diag['extrap_rate']:.1f}% 的测试样本超出训练集 95% 置信包络半径。
3. **样本规模因素（影响度评分 {diag['radar_scores']['③ 样本规模约束']}）**：训练集共 {diag['n_train']} 环，激活 {diag['active_factors']}/7 个无量纲因子，L1 正则化收缩效应对极端峰谷值产生适度平滑。
4. **线性假设因素（影响度评分 {diag['radar_scores']['④ 线性假设截断']}）**：二阶非线性残差相关度 r = {diag['quad_r']:.2f}，加性线性假设忽略了软硬交界处的高阶耦合项。
5. **无量纲因子机制覆盖（影响度评分 {diag['radar_scores']['⑤ 因子机制未覆盖度']}）**：预测偏小占比 {diag['under_rate']:.1f}%，偏大占比 {diag['over_rate']:.1f}%；预测显著偏小环指示存在盾壳外周摩阻力剧增、结泥饼或刀具磨损等未覆盖机制。
6. **地质突变因素（影响度评分 {diag['radar_scores']['⑥ 地质刚度突变']}）**：相邻环弹性模量突变梯度与预测误差相关系数 r = {diag['jump_r']:.2f}，地层刚度骤增引发瞬态推力冲击偏差。
"""
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "📝 下载完整建模与六维归因报告 (.md)",
        data=report_md.encode('utf-8-sig'),
        file_name="TBM_Intelligent_Agent_Report.md",
        mime="text/markdown",
        use_container_width=True
    )
    dl2.download_button(
        "📊 下载测试集逐环对比与偏差定位表 (.csv)",
        data=diag["detail_df"].to_csv(index=False).encode('utf-8-sig'),
        file_name="TBM_Test_Predictions_and_Diagnosis.csv",
        mime="text/csv",
        use_container_width=True
    )