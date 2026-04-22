"""
Federated Learning Experiment: IID vs Non-IID on MNIST
=======================================================

Experiment 1: IID vs Non-IID convergence comparison (10 clients)
Experiment 2: Effect of number of clients (5, 10, 20) under Non-IID

Run:
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data import load_data, iid_split, noniid_split
from model import MLP
from client import Client
from server import Server
from optimizers import SGD


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ROUNDS       = 50
LOCAL_EPOCHS = 5      # more local steps = more client drift = visible gap
LR           = 0.01   # lower lr works better with 5 local epochs


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_data()
print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")


# ──────────────────────────────────────────────
# Helper: build and run one FL experiment
# ──────────────────────────────────────────────
def run_experiment(client_data, rounds=ROUNDS, lr=LR, epochs=LOCAL_EPOCHS, label=""):

    # Verify what each client sees (printed once per experiment)
    print(f"  Client class distribution:")
    for i, (X, y) in enumerate(client_data):
        classes = np.unique(y).tolist()
        print(f"    Client {i}: {len(X)} samples, classes {classes}")

    global_model = MLP()
    server = Server(global_model)

    clients = []
    for X, y in client_data:
        local_model = MLP()
        local_model.set_weights(global_model.get_weights())
        clients.append(Client(X, y, local_model, SGD(lr=lr), epochs=epochs))

    acc_history = []

    for r in range(rounds):
        # Each client trains locally for `epochs` steps
        updates = [client.train() for client in clients]

        # Server aggregates with FedAvg
        new_weights = server.aggregate(updates)

        # Broadcast updated global weights back to every client
        for client in clients:
            client.model.set_weights(new_weights)

        acc = server.model.accuracy(X_test, y_test)
        acc_history.append(acc)

        if (r + 1) % 10 == 0:
            print(f"  [{label}] Round {r+1:>2}/{rounds}  Accuracy: {acc:.4f}")

    return acc_history


# ──────────────────────────────────────────────
# Experiment 1: IID vs Non-IID (10 clients)
# ──────────────────────────────────────────────
N_CLIENTS = 10
print("=" * 52)
print(f"Experiment 1: IID vs Non-IID  ({N_CLIENTS} clients)")
print("=" * 52)

print("\nRunning IID...")
iid_data = iid_split(X_train, y_train, N_CLIENTS)
acc_iid  = run_experiment(iid_data, label="IID")

print("\nRunning Non-IID...")
noniid_data = noniid_split(X_train, y_train, N_CLIENTS)
acc_noniid  = run_experiment(noniid_data, label="Non-IID")


# ──────────────────────────────────────────────
# Experiment 2: Effect of #clients under Non-IID
# Non-IID here means each client gets one class,
# so we use iid_split but restrict to n subsets
# and vary the count to show scalability effect
# ──────────────────────────────────────────────
client_counts = [5, 10, 20]
noniid_by_clients = {}

print(f"\n{'=' * 52}")
print("Experiment 2: Number of clients (Non-IID)")
print("=" * 52)

for n in client_counts:
    print(f"\nRunning Non-IID, {n} clients...")
    # For n != 10, assign one class per client cycling through digits
    data = []
    for i in range(n):
        cls = i % 10
        idx = np.where(y_train == cls)[0]
        # if n > 10, split same class across multiple clients
        chunk = np.array_split(idx, n // 10 + 1)[i // 10]
        data.append((X_train[chunk], y_train[chunk]))
    noniid_by_clients[n] = run_experiment(data, label=f"{n} clients")


# ──────────────────────────────────────────────
# Gap analysis
# ──────────────────────────────────────────────
gap = [iid - noniid for iid, noniid in zip(acc_iid, acc_noniid)]
max_gap       = max(gap)
max_gap_round = gap.index(max_gap) + 1
final_gap     = acc_iid[-1] - acc_noniid[-1]

print(f"\n{'=' * 52}")
print("GAP ANALYSIS (IID vs Non-IID)")
print("=" * 52)
print(f"  Max accuracy gap:   {max_gap:.4f}  (at round {max_gap_round})")
print(f"  Final accuracy gap: {final_gap:.4f}  (at round {ROUNDS})")


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
rounds_axis = list(range(1, ROUNDS + 1))

fig = plt.figure(figsize=(14, 5))
fig.suptitle(
    "Federated Learning on MNIST — Convergence Analysis",
    fontsize=13, fontweight="bold"
)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Plot 1: IID vs Non-IID ──
ax1 = fig.add_subplot(gs[0])
ax1.plot(rounds_axis, acc_iid,
         label="IID", color="#2563eb", linewidth=2.0, marker="o", markersize=3)
ax1.plot(rounds_axis, acc_noniid,
         label="Non-IID", color="#dc2626", linewidth=2.0,
         marker="s", markersize=3, linestyle="--")
ax1.fill_between(rounds_axis, acc_iid, acc_noniid,
                 alpha=0.12, color="#dc2626", label="Gap")

ax1.set_title(f"IID vs Non-IID ({N_CLIENTS} clients, {LOCAL_EPOCHS} local epochs)",
              fontsize=10)
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Test Accuracy")
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotate final values
offset = 0.04
ax1.annotate(
    f"IID: {acc_iid[-1]:.3f}",
    xy=(ROUNDS, acc_iid[-1]),
    xytext=(ROUNDS - 18, acc_iid[-1] + offset),
    fontsize=9, color="#2563eb",
    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1)
)
ax1.annotate(
    f"Non-IID: {acc_noniid[-1]:.3f}",
    xy=(ROUNDS, acc_noniid[-1]),
    xytext=(ROUNDS - 18, acc_noniid[-1] - offset - 0.05),
    fontsize=9, color="#dc2626",
    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1)
)

# ── Plot 2: Effect of #clients ──
ax2 = fig.add_subplot(gs[1])
colors = ["#7c3aed", "#dc2626", "#ea580c"]

for (n, acc), color in zip(noniid_by_clients.items(), colors):
    ax2.plot(rounds_axis, acc,
             label=f"{n} clients", color=color,
             linewidth=2.0, marker="o", markersize=3)

ax2.set_title("Effect of Number of Clients (Non-IID)", fontsize=11)
ax2.set_xlabel("Communication Round")
ax2.set_ylabel("Test Accuracy")
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to results.png")


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────
print("\n" + "=" * 58)
print("RESULTS SUMMARY")
print("=" * 58)
print(f"{'Setting':<30} {'Final Acc':>10} {'Round 10 Acc':>13}")
print("-" * 58)
print(f"{'IID (10 clients)':<30} {acc_iid[-1]:>10.4f} {acc_iid[9]:>13.4f}")
print(f"{'Non-IID (10 clients)':<30} {acc_noniid[-1]:>10.4f} {acc_noniid[9]:>13.4f}")
for n, acc in noniid_by_clients.items():
    lbl = f"Non-IID ({n} clients)"
    print(f"{lbl:<30} {acc[-1]:>10.4f} {acc[9]:>13.4f}")

print(f"\nMax IID-NonIID gap : {max_gap:.4f}  at round {max_gap_round}")
print(f"Final gap          : {final_gap:.4f}")