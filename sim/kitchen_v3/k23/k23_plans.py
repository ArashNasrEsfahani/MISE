"""K23 plan helpers shared by the capture, the supervisor decisions and the layout check (one source of truth).

PARITY (Arash 2026-10-17, ledger 'ARASH DECISION: size parity (b-150) with close-up captures'; Astra K2-PARITY-r1):
up to 150 extra TRAIN episodes with the frozen close-up view policy, episodes 4000+i, protocol PARITY_PROTOCOL, setup seed
stable_seed(scene, PARITY_PROTOCOL, i) (the frozen view RNG keeps stable_seed(scene, 'views', episode))."""
PARITY_PROTOCOL = "parity-close-train-v1"
PARITY_MAX = 150
PARITY_BASE = 4000


def parity_plan(scene: str, kd, n: int = PARITY_MAX) -> list:
    return [{"episode": PARITY_BASE + i, "split": "train", "seed": kd.stable_seed(scene, PARITY_PROTOCOL, i),
             "protocol": PARITY_PROTOCOL} for i in range(n)]
