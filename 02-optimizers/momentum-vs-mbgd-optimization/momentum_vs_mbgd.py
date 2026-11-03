
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 0. HYPERPARAMETERS (change these freely)

SEED          = 42
N_SAMPLES     = 256
CORRELATION   = 0.98      # how strongly feature 1 and feature 2 move together
NOISE_STD     = 0.5
TRUE_W        = np.array([2.0, 1.0])
TRUE_B        = 0.5
LEARNING_RATE = 0.40
BATCH_SIZE    = 32
EPOCHS        = 25
BETA          = 0.9       # momentum coefficient (0 -> plain MBGD)
INIT_W        = np.array([-1.0, 2.0])
INIT_B        = 0.0



# 1. DATA
def create_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, CORRELATION], [CORRELATION, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, size=N_SAMPLES)
    X = (X - X.mean(axis=0)) / X.std(axis=0)           # standardize
    y = X @ TRUE_W + TRUE_B + rng.normal(0, NOISE_STD, N_SAMPLES)
    return X, y


def create_mini_batches(n_samples, batch_size, n_epochs, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    all_epochs = []
    for _ in range(n_epochs):
        idx = rng.permutation(n_samples)
        all_epochs.append([idx[i:i + batch_size] for i in range(0, n_samples, batch_size)])
    return all_epochs



# 2. MODEL, LOSS, GRADIENTS
def predict(X, w, b):
    return X @ w + b                                    # y_hat = Xw + b


def compute_loss(X, y, w, b):
    err = predict(X, w, b) - y
    return np.mean(err ** 2)                            # MSE


def compute_gradients(X, y, w, b):
    m = len(y)
    err = predict(X, w, b) - y
    grad_w = (2.0 / m) * (X.T @ err)
    grad_b = (2.0 / m) * np.sum(err)
    return grad_w, grad_b


# 3. TRAINING LOOPS
def train_mbgd(X, y, batches, lr):
    w, b = INIT_W.copy(), float(INIT_B)
    hist = {"loss": [compute_loss(X, y, w, b)],
            "path": [[*w, b]], 
            "grad_norm": [], 
            "step_norm": []}

    for epoch_batches in batches:
        for idx in epoch_batches:
            gw, gb = compute_gradients(X[idx], y[idx], w, b)

            #update the parameters using the gradients and learning rate
            step_w, step_b = lr * gw, lr * gb
            w = w - step_w
            b = b - step_b
        

            hist["path"].append([*w, b])
            hist["grad_norm"].append(np.linalg.norm([*gw, gb]))
            hist["step_norm"].append(np.linalg.norm([*step_w, step_b]))
        hist["loss"].append(compute_loss(X, y, w, b))    # full-data loss per epoch
    return w, b, hist


def train_mbgd_momentum(X, y, batches, lr, beta):
    w, b = INIT_W.copy(), float(INIT_B)
    v_w, v_b = np.zeros_like(w), 0.0                    # velocity starts at zero
    hist = {"loss": [compute_loss(X, y, w, b)],
            "path": [[*w, b]], 
            "grad_norm": [], 
            "step_norm": [],
            "vel_norm": []}

    for epoch_batches in batches:
        for idx in epoch_batches:
            gw, gb = compute_gradients(X[idx], y[idx], w, b)

            v_w = beta * v_w + (1 - beta) * gw
            v_b = beta * v_b + (1 - beta) * gb

            step_w, step_b = lr * v_w, lr * v_b
            w = w - step_w
            b = b - step_b

            hist["path"].append([*w, b])
            hist["grad_norm"].append(np.linalg.norm([*gw, gb]))
            hist["vel_norm"].append(np.linalg.norm([*v_w, v_b]))
            hist["step_norm"].append(np.linalg.norm([*step_w, step_b]))
        hist["loss"].append(compute_loss(X, y, w, b))
    return w, b, hist



# 4. PLOTS
def plot_loss(h1, h2, beta):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, log in zip(ax, [False, True]):
        a.plot(h1["loss"], "o-", label="MBGD", color="tab:blue")
        a.plot(h2["loss"], "s-", label=f"MBGD + Momentum (beta={beta})", color="tab:red")
        a.set_xlabel("Epoch"); a.set_ylabel("MSE on full dataset")
        a.grid(alpha=.3); a.legend()
        if log: a.set_yscale("log"); a.set_title("Loss vs Epoch (log scale)")
        else:   a.set_title("Loss vs Epoch")
    plt.tight_layout(); plt.savefig("plot1_loss.png", dpi=130)


def plot_trajectory(X, y, h1, h2):
    p1, p2 = np.array(h1["path"]), np.array(h2["path"])

    # --- (a) parameters vs iteration ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, name in enumerate(["w1", "w2"]):
        ax[k].plot(p1[:, k], label="MBGD", color="tab:blue")
        ax[k].plot(p2[:, k], label="Momentum", color="tab:red")
        ax[k].axhline(TRUE_W[k], color="k", ls="--", lw=.8, label="true value")
        ax[k].set_xlabel("Iteration (mini-batch update)"); ax[k].set_ylabel(name)
        ax[k].set_title(f"{name} over time"); ax[k].grid(alpha=.3); ax[k].legend()
    plt.tight_layout(); plt.savefig("plot2_parameters.png", dpi=130)

    # --- (b) contour of the loss surface with both paths ---
    # Bias is fixed at its optimal value so we can draw a clean 2-D slice.
    b_fix = y.mean()
    lo = np.minimum(p1[:, :2].min(0), p2[:, :2].min(0)) - 0.7
    hi = np.maximum(p1[:, :2].max(0), p2[:, :2].max(0)) + 0.7
    g1, g2 = np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200)
    G1, G2 = np.meshgrid(g1, g2)
    Z = np.array([[compute_loss(X, y, np.array([a, c]), b_fix) for a in g1] for c in g2])

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contour(G1, G2, Z, levels=np.geomspace(Z.min() + 0.05, Z.max(), 25), cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax.plot(p1[:, 0], p1[:, 1], "o-", ms=3, lw=1.2, color="tab:blue", label="MBGD")
    ax.plot(p2[:, 0], p2[:, 1], "s-", ms=3, lw=1.2, color="tab:red", label="MBGD + Momentum")
    ax.plot(*p1[0, :2], "k^", ms=11, label="start")
    ax.plot(*TRUE_W, "g*", ms=16, label="true weights")
    ax.set_xlabel("w1"); ax.set_ylabel("w2")
    ax.set_title("Loss surface (bias fixed) and optimization paths")
    ax.legend(); plt.tight_layout(); plt.savefig("plot3_contour.png", dpi=130)


def plot_velocity(h1, h2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h1["grad_norm"], color="tab:blue", alpha=.8, label="|gradient| (MBGD)")
    ax[0].plot(h2["grad_norm"], color="tab:red", alpha=.8, label="|gradient| (Momentum run)")
    ax[0].plot(h2["vel_norm"], color="k", lw=2, label="|velocity| (Momentum)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("magnitude")
    ax[0].set_title("Noisy gradients vs smooth velocity"); ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(h1["step_norm"], color="tab:blue", label="step size MBGD")
    ax[1].plot(h2["step_norm"], color="tab:red", label="step size Momentum")
    ax[1].set_yscale("log"); ax[1].set_xlabel("Iteration"); ax[1].set_ylabel("|parameter update|")
    ax[1].set_title("Size of each parameter update"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("plot4_velocity.png", dpi=130)


def zigzag_score(path):
    """Fraction of consecutive steps whose direction reverses (dot product < 0)."""
    d = np.diff(np.array(path)[:, :2], axis=0)
    return np.mean(np.sum(d[1:] * d[:-1], axis=1) < 0)



# 5. MAIN
def main():
    X, y = create_dataset()
    batches = create_mini_batches(N_SAMPLES, BATCH_SIZE, EPOCHS)   # shared by both runs

    H = 2 * np.cov(X.T)
    print("Hessian eigenvalues:", np.round(np.linalg.eigvalsh(H), 3),
          "-> plain GD is stable only if lr <", round(2 / np.linalg.eigvalsh(H).max(), 3))
    print(f"Chosen lr = {LEARNING_RATE}\n")

    w1, b1, h1 = train_mbgd(X, y, batches, LEARNING_RATE)
    w2, b2, h2 = train_mbgd_momentum(X, y, batches, LEARNING_RATE, BETA)

    print("Epoch | MBGD Loss | Momentum Loss")
    for e in range(EPOCHS + 1):
        print(f"{e:5d} | {h1['loss'][e]:9.4f} | {h2['loss'][e]:9.4f}")

    noise_floor = NOISE_STD ** 2
    print("\n=========== FINAL RESULTS ===========")
    print(f"Epochs: {EPOCHS} | Learning rate: {LEARNING_RATE} | Batch size: {BATCH_SIZE} | Beta: {BETA}")
    print(f"(Irreducible noise level of the data ~ {noise_floor:.3f})")
    print(f"MBGD     final loss: {h1['loss'][-1]:.4f}   w={np.round(w1,3)}  b={b1:.3f}")
    print(f"Momentum final loss: {h2['loss'][-1]:.4f}   w={np.round(w2,3)}  b={b2:.3f}")
    print(f"True parameters:                 w={TRUE_W}  b={TRUE_B}")
    print(f"Zig-zag score (direction reversals) MBGD: {zigzag_score(h1['path']):.2f} | "
          f"Momentum: {zigzag_score(h2['path']):.2f}")

    pd.DataFrame({"epoch": range(EPOCHS + 1), "mbgd_loss": h1["loss"],
                  "momentum_loss": h2["loss"]}).to_csv("loss_table.csv", index=False)

    plot_loss(h1, h2, BETA); plot_trajectory(X, y, h1, h2); plot_velocity(h1, h2)
    plt.show()


if __name__ == "__main__":
    main()