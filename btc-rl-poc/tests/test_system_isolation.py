"""SYSTEM ISOLATION guard — the live trading desk and the sealed-research program
must stay COMPLETELY DISCONNECTED so a change in one cannot affect the other.

Architecture (see SYSTEM_BOUNDARY.md):
  shared capture substrate  btc_rl.online freezes each KXBTC15M window at T0 and
                            records exact-BRTI settlement — the market feed both
                            systems read. It is infrastructure, not a "system".
  LIVE trading desk         Follower/Oracle arms, paper P&L, bankroll (home page).
  RESEARCH program          sealed TRUE15M Oracle: offline models, TEST_V2, firewall.

Isolation rules enforced here:
  1. LIVE source references NO research token (research changes can't reach the desk).
  2. RESEARCH source references NO live-desk PRIVATE file (models/ledgers/P&L).
  3. No cross-imports between the two code sets.
  4. The ONLY live->research touch is the research narrator READING the shared capture
     heartbeat (online_status.json, kalshi_binary_log.jsonl) — read-only, allowlisted.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
BTC = ROOT / "btc_rl"

# ---- the two code sets ----
LIVE_SOURCES = [
    BTC / "online.py",
    SCRIPTS / "emit_live_desk.py",
    SCRIPTS / "emit_home_snapshot.py",
    SCRIPTS / "emit_traders_snapshot.py",
]
RESEARCH_SOURCES = [
    BTC / "test_v2_firewall.py",
    BTC / "research_events.py",
    SCRIPTS / "true15m_offline_program.py",
    SCRIPTS / "true15m_contract_inventory.py",
    SCRIPTS / "loss_calibration_sweeps.py",
    SCRIPTS / "model_failure_diagnostics.py",
    SCRIPTS / "distributional_models.py",
    SCRIPTS / "sealed_test_governance.py",
    SCRIPTS / "test_v2_capture_audit.py",
    SCRIPTS / "test_v2_power_analysis.py",
    SCRIPTS / "freeze_feature_universe.py",
    SCRIPTS / "coarse_feature_factory.py",
    SCRIPTS / "fine_feature_factory.py",
]

# tokens that name RESEARCH-only artifacts/modules
RESEARCH_TOKENS = ["true15m", "TEST_V2", "SEALED_TEST", "FEATURE_UNIVERSE",
                   "LOSS_CALIBRATION", "MODEL_FAILURE_DIAGNOSTICS", "test_v2_firewall",
                   "loss_calibration_sweeps", "distributional_models"]
# files PRIVATE to the live desk (its models / ledgers / P&L) — research must never touch
LIVE_PRIVATE_TOKENS = ["pt_trades", "pt2_trades", "pt3_trades", "pt6_trades", "pt8_trades",
                       "live_desk.json", "home_snapshot", "pt6_logit", "q_table_online",
                       "demo_orders", "demo_account", "agent_recommendations"]
# the shared capture heartbeat the research narrator may READ (read-only, by design)
SHARED_CAPTURE_READ_ALLOW = ["online_status.json", "kalshi_binary_log.jsonl"]


def _text(p):
    return p.read_text() if p.exists() else ""


@pytest.mark.parametrize("src", LIVE_SOURCES, ids=lambda p: p.name)
def test_live_source_has_no_research_dependency(src):
    body = _text(src)
    hits = [t for t in RESEARCH_TOKENS if t in body]
    assert not hits, f"{src.name} references research tokens {hits} — live desk must not depend on research"


@pytest.mark.parametrize("src", RESEARCH_SOURCES, ids=lambda p: p.name)
def test_research_source_never_touches_live_private_files(src):
    body = _text(src)
    hits = [t for t in LIVE_PRIVATE_TOKENS if t in body]
    assert not hits, f"{src.name} references live-desk private files {hits} — research must stay disconnected"


def test_no_cross_imports():
    for src in RESEARCH_SOURCES:
        b = _text(src)
        assert "btc_rl.online" not in b and "import online" not in b, \
            f"{src.name} imports the live trading daemon"
    for src in LIVE_SOURCES:
        b = _text(src)
        for mod in ("test_v2_firewall", "research_events"):
            assert mod not in b, f"{src.name} imports research module {mod}"


def test_shared_capture_touch_is_readonly_and_allowlisted():
    """The research narrator's ONLY live-side reference is the shared capture heartbeat,
    and it is read-only (never opened for write)."""
    snap = SCRIPTS / "emit_research_snapshot.py"
    body = _text(snap)
    # it must not reference any live-desk PRIVATE file
    assert not [t for t in LIVE_PRIVATE_TOKENS if t in body], \
        "research snapshot references a live-desk private file"
    # the shared heartbeat files may be READ but never WRITTEN from the research side
    for bad in ('online_status.json").write', 'kalshi_binary_log.jsonl").write',
                'R / "online_status.json", "w', 'R / "kalshi_binary_log.jsonl", "w'):
        assert bad not in body, f"research snapshot appears to WRITE a shared/live file ({bad})"
