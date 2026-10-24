"""Pure Phase-6a wiring self-test.

Runs without starting AI2-THOR. It stubs the import-only ai2thor.controller module,
imports webapp, and verifies the rating/reward path against a temporary sqlite DB.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types


def _stub_ai2thor():
    ai2thor = types.ModuleType("ai2thor")
    controller = types.ModuleType("ai2thor.controller")

    class Controller:  # import-time placeholder only; never instantiated here
        pass

    controller.Controller = Controller
    ai2thor.controller = controller
    sys.modules.setdefault("ai2thor", ai2thor)
    sys.modules.setdefault("ai2thor.controller", controller)


def main():
    _stub_ai2thor()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    import knowledge
    import memory
    import webapp

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "phase6a.db")
    scene = "FloorPlan1"
    command = "pick up the apple"

    mem = memory.Memory(path=db)
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "kb.json"))
    webapp._memory = mem
    webapp._kb = kb
    webapp._feedback = None
    webapp._bandit = None
    webapp._bandit_mod = None

    dec = webapp._decide_plan_strategy(command, scene, memory_block="known Apple location")
    did = int(dec["decision_id"])
    assert did > 0, f"expected a logged bandit decision, got {dec!r}"

    with webapp._lock:
        webapp._state["scene"] = scene
        webapp._state["cur_cmd"] = command
        webapp._state["plan"] = {"decision_id": did, "command": command, "steps": [{"status": "failed"}]}

    mem.log_episode(scene=scene, command=command, level="L1", action="pickup",
                    args={"object": "Apple"}, success=False, error="not visible")
    env_reward = webapp._reward_plan(did, scene, command)
    assert env_reward == 0.0, f"failed episode should produce 0.0 env reward, got {env_reward!r}"

    row = webapp._get_bandit().conn.execute(
        "SELECT reward, source FROM bandit_decisions WHERE id=?", (did,)).fetchone()
    assert row["reward"] == 0.0 and row["source"] == "environment", dict(row)

    rated = webapp._rate_plan(did, 1)
    assert rated["ok"] and rated["rows"] == 1 and rated["reward"] == 1.0, rated

    row = webapp._get_bandit().conn.execute(
        "SELECT reward, source FROM bandit_decisions WHERE id=?", (did,)).fetchone()
    assert row["reward"] == 1.0 and row["source"] == "human", dict(row)
    ep = mem.conn.execute("SELECT rating FROM episodes ORDER BY id DESC LIMIT 1").fetchone()
    assert ep["rating"] == 1.0, dict(ep)

    bad = webapp._rate_plan(did, 0)
    assert not bad["ok"], bad

    webapp._get_bandit().close()
    mem.close()
    print("PHASE6A SELFTEST OK")


if __name__ == "__main__":
    main()
