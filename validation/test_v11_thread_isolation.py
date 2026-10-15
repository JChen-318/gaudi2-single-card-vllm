"""Engine thread isolation (VLLM_HPU_PIN_MAIN_CPU) with the OS affinity calls replaced by fakes."""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro = v6_support._load("v1/worker/kiro_v11.py", "kiro_v11_isolation")


class FakeOS:

    def __init__(self, monkeypatch, tids, allowed=frozenset(range(8))):
        self.affinity = {}
        self.tids = list(tids)
        self.allowed = set(allowed)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(self.allowed), raising=False)
        monkeypatch.setattr(os, "sched_setaffinity", self.set, raising=False)
        real_listdir = os.listdir
        monkeypatch.setattr(os, "listdir",
                            lambda path: [str(t) for t in self.tids] if path == "/proc/self/task" else real_listdir(path))

    def set(self, pid, cpus):
        self.affinity[pid] = set(cpus)


def test_pins_main_and_moves_every_other_thread_off_the_core(monkeypatch):
    main = threading.get_native_id()
    fake = FakeOS(monkeypatch, [main, 101, 102])
    iso = kiro.ThreadIsolation(2)
    monkeypatch.setattr(iso, "_siblings", lambda cpu: {2, 6})
    iso.apply()
    assert fake.affinity[0] == {2}
    assert fake.affinity[101] == fake.affinity[102] == {0, 1, 3, 4, 5, 7}
    assert main not in fake.affinity
    # threads created later are moved off by refresh; known ones are left alone
    fake.tids.append(103)
    fake.affinity.clear()
    iso.refresh()
    assert fake.affinity == {103: {0, 1, 3, 4, 5, 7}}
    iso.apply()  # idempotent
    assert 0 not in fake.affinity


def test_unusable_cpu_is_skipped_without_pinning(monkeypatch):
    fake = FakeOS(monkeypatch, [threading.get_native_id(), 101], allowed={0, 1})
    iso = kiro.ThreadIsolation(5)
    iso.apply()
    assert iso.failed and fake.affinity == {}
    iso.refresh()
    assert fake.affinity == {}
