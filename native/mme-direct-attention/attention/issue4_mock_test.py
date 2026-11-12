# SPDX-License-Identifier: MIT
"""GitHub issue #4 (gaudi2-single-card-vllm): CPU-only check of _Plan.set_valid failure handling.

Runs the real _Plan.set_valid source (extracted with ast, as in the issue's reproduction) against mocked native
updates; needs no Gaudi, torch or the bridge. Run: python issue4_mock_test.py
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).with_name("h3_direct_attention.py")
tree = ast.parse(SRC.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Plan")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_valid")
ns: dict = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SRC), "exec"), ns)
set_valid = ns["set_valid"]


class PartialNative:
    """The issue's model: the native update fails after slot 0 changed (pre-fix native behaviour)."""

    def __init__(self):
        self.valid = 37750
        self.slots = [37750, 37750]
        self.calls = 0

    def _set_valid(self, valid):
        self.calls += 1
        if self.slots == [valid, valid]:
            return 0
        self.slots[0] = valid
        return -5


class AtomicNative(PartialNative):
    """The fixed native side: a failing update is undone, a later call for the current length is a no-op."""

    def __init__(self, fail_next=1):
        super().__init__()
        self.fail_next = fail_next

    def _set_valid(self, valid):
        self.calls += 1
        if self.slots == [valid, valid]:
            return 0
        if self.fail_next:
            self.fail_next -= 1
            return -5                      # slots restored: unchanged
        self.slots = [valid, valid]
        return 0


failures = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failures.append(name)


# Issue scenario: a True result must never be returned from a stale Python length.
p = PartialNative()
check("partial: change fails", set_valid(p, 37760) is False)
check("partial: length forgotten", p.valid is None)
r = set_valid(p, 37750)
check("partial: old length calls native again", p.calls == 2)
check("partial: True only if slots agree", r is not True or p.slots == [37750, 37750])

# Fixed native: failure then old length -> consistent and accepted; new length works on retry.
p = AtomicNative()
check("atomic: change fails", set_valid(p, 37760) is False and p.slots == [37750, 37750])
check("atomic: old length accepted after native check", set_valid(p, 37750) is True and p.calls == 2)
check("atomic: old length cached afterwards", set_valid(p, 37750) is True and p.calls == 2)
check("atomic: new length succeeds", set_valid(p, 37760) is True and p.slots == [37760, 37760] and p.valid == 37760)

# Poisoned plan (restore failed): every call fails, the caller rebuilds.
p = AtomicNative(fail_next=10**9)
check("poisoned: change fails", set_valid(p, 37760) is False)
p.slots = [None, None]
check("poisoned: old length fails too", set_valid(p, 37750) is False and p.valid is None)

sys.exit(1 if failures else 0)
