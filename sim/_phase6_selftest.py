"""Pure Phase-6 human-feedback wiring self-test.

No AI2-THOR controller is started. The test stubs ai2thor at import time and
uses temporary sqlite/JSON stores to exercise the webapp helper layer.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types


def _stub_ai2thor():
    ai2thor = types.ModuleType("ai2thor")
    controller = types.ModuleType("ai2thor.controller")

    class Controller:
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
    mem = memory.Memory(path=os.path.join(tmp, "phase6.db"))
    kb = knowledge.KnowledgeBase(path=os.path.join(tmp, "kb.json"))
    webapp._memory = mem
    webapp._kb = kb
    webapp._feedback = None
    webapp._bandit = None
    webapp._bandit_mod = None

    scene = "FloorPlan1"
    command = "pick up the apple"
    with webapp._lock:
        webapp._state["scene"] = scene
        webapp._state["cur_cmd"] = command
        webapp._state["jpeg"] = b"not-a-real-jpeg-but-good-enough-for-label-storage"
        webapp._state["held"] = "none"
        webapp._state["objects"] = [{"type": "Apple", "actions": ["Pick Up"], "visible": True}]

    repaired, repairs = webapp._repair_plan_for_command(command, [{"action": "go_to", "object": "Apple"}])
    assert repaired == [{"action": "go_to", "object": "Apple"},
                        {"action": "pickup", "object": "Apple", "to": None, "arg": None}], repaired
    assert repairs == ["added pickup Apple"], repairs
    assert webapp._pickup_target("drop the apple and pick up bread") == "Bread"

    fact = webapp._store_human_teaching("fact: mugs can be inside cabinets")
    assert fact["ok"] and fact["kind"] == "fact", fact
    pref = webapp._store_human_teaching("when i say drop something, i generally mean counter top. NOT on floor.")
    assert pref["ok"] and pref["kind"] == "fact", pref
    spatial = webapp._store_human_teaching("spatial: Mug in Cabinet")
    assert spatial["ok"] and spatial["kind"] == "spatial", spatial
    belief = mem.query_location(scene, "Mug", track="perceived")
    assert belief and belief["receptacle"] == "Cabinet" and belief["source"] == "human", belief

    webapp._start_demo("put the mug in the sink")
    webapp._record_demo_step({"action": "go_to", "object": "Mug"})
    webapp._record_demo_step({"action": "put", "to": "Sink"})
    demo = webapp._stop_demo()
    assert demo["ok"] and demo["saved"] and demo["steps"] == 2, demo
    assert kb.data["procedures"][-1]["source"] == "demo", kb.data["procedures"][-1]

    dec = webapp._decide_plan_strategy(command, scene, memory_block="Apple @ CounterTop")
    did = int(dec["decision_id"])
    mem.log_episode(scene=scene, command=command, level="L1", action="pickup",
                    args={"object": "Apple"}, success=True, error=None)
    with webapp._lock:
        webapp._state["plan"] = {"decision_id": did, "command": command,
                                 "raw_steps": [{"action": "pickup", "object": "Apple"}],
                                 "steps": [{"status": "done"}]}
    rated = webapp._rate_plan(did, 1)
    assert rated["ok"] and rated["rows"] == 1, rated
    assert kb.data["procedures"][-1]["source"] == "human", kb.data["procedures"][-1]

    # _store_perception_feedback writes the current frame to <HERE>/captures/; point it at the temp dir so this
    # test's fake 49-byte "jpeg" never lands in the real capture library (it showed as broken thumbnails)
    real_here = webapp.HERE
    webapp.HERE = tmp
    try:
        pf = webapp._store_perception_feedback("Apple", "CounterTop", "1,2,3,4")
    finally:
        webapp.HERE = real_here
    assert not os.path.exists(os.path.join(real_here, "captures", os.path.basename(pf.get("image_path") or "x.jpg"))) \
        or os.path.getsize(os.path.join(real_here, "captures", os.path.basename(pf.get("image_path") or "x.jpg"))) > 64, pf
    assert pf["ok"] and pf["box"]["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0], pf
    belief = mem.query_location(scene, "Apple", track="perceived")
    assert belief and belief["receptacle"] == "CounterTop" and belief["source"] == "human", belief

    webapp._get_bandit().close()
    mem.close()
    print("PHASE6 SELFTEST OK")


if __name__ == "__main__":
    main()
