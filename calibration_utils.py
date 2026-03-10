# -*- coding: utf-8 -*-
"""
FMS 재보정(직관 기반) 보조 유틸리티.

- 데이터 스냅샷 저장/로드 (작업 중 데이터 고정)
- 인터랙티브 병합 정렬(merge sort) 상태 머신
- 세션 저장/로드 (중단 후 재개)

주의: "정답"은 순서(rank)이며 점수 간격이 아님.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SNAPSHOT_ROOT_DIR = "fms_calibration_snapshots"
SESSION_ROOT_DIR = "fms_calibration_sessions"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso_local() -> str:
    # Streamlit 앱은 보통 KST를 쓰지만, 여기서는 호출 측에서 원하는 타임존 문자열을 넣는 것을 권장.
    return datetime.now().isoformat(timespec="seconds")


def create_snapshot_id(prefix: str = "snap") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def snapshot_dir(snapshot_id: str) -> str:
    return os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id)


def save_snapshot(
    snapshot_id: str,
    *,
    prices_krw: pd.DataFrame,
    ohlc_data: Optional[pd.DataFrame],
    meta: Dict[str, Any],
) -> str:
    """
    스냅샷을 디스크에 저장합니다.
    - prices_krw: wide DF (index=date, columns=symbol)
    - ohlc_data: MultiIndex columns인 DF를 포함할 수 있어 pickle로 저장
    - meta: JSON 직렬화 가능한 딕셔너리
    """
    root = snapshot_dir(snapshot_id)
    _ensure_dir(root)

    prices_path = os.path.join(root, "prices_krw.pkl")
    prices_krw.to_pickle(prices_path)

    ohlc_path = os.path.join(root, "ohlc.pkl")
    if ohlc_data is not None:
        ohlc_data.to_pickle(ohlc_path)
    else:
        # 명시적으로 "없음"을 기록
        with open(ohlc_path, "wb") as f:
            f.write(b"")

    meta_path = os.path.join(root, "meta.json")
    meta_out = dict(meta)
    meta_out.setdefault("snapshot_id", snapshot_id)
    meta_out.setdefault("created_at", now_iso_local())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    return root


def load_snapshot(snapshot_id: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    root = snapshot_dir(snapshot_id)
    prices_path = os.path.join(root, "prices_krw.pkl")
    ohlc_path = os.path.join(root, "ohlc.pkl")
    meta_path = os.path.join(root, "meta.json")

    prices_krw = pd.read_pickle(prices_path)

    ohlc_data: Optional[pd.DataFrame] = None
    if os.path.exists(ohlc_path) and os.path.getsize(ohlc_path) > 0:
        ohlc_data = pd.read_pickle(ohlc_path)

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return prices_krw, ohlc_data, meta


def list_snapshots() -> List[str]:
    if not os.path.exists(SNAPSHOT_ROOT_DIR):
        return []
    snaps = []
    for name in os.listdir(SNAPSHOT_ROOT_DIR):
        p = os.path.join(SNAPSHOT_ROOT_DIR, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
            snaps.append(name)
    snaps.sort(reverse=True)
    return snaps


def session_path(session_id: str) -> str:
    _ensure_dir(SESSION_ROOT_DIR)
    return os.path.join(SESSION_ROOT_DIR, f"{session_id}.json")


def save_session(session_id: str, state: Dict[str, Any]) -> str:
    path = session_path(session_id)
    state_out = dict(state)
    state_out.setdefault("session_id", session_id)
    state_out["saved_at"] = now_iso_local()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state_out, f, ensure_ascii=False, indent=2)
    return path


def load_session(session_id: str) -> Dict[str, Any]:
    path = session_path(session_id)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> List[str]:
    if not os.path.exists(SESSION_ROOT_DIR):
        return []
    out = []
    for fn in os.listdir(SESSION_ROOT_DIR):
        if fn.endswith(".json"):
            out.append(fn[:-5])
    out.sort(reverse=True)
    return out


def mergesort_estimated_max_comparisons(n: int) -> int:
    """
    병합정렬의 비교 횟수 상한(최악)을 반환합니다.
    참고: 실제 비교 횟수는 조기 소진으로 더 작을 수 있음.
    """
    if n <= 1:
        return 0
    # n*ceil(log2 n) - 2^ceil(log2 n) + 1
    import math

    k = math.ceil(math.log2(n))
    return n * k - (2**k) + 1


@dataclass
class MergeState:
    left: List[str]
    right: List[str]
    i: int = 0
    j: int = 0
    out: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.out is None:
            self.out = []

    def current_pair(self) -> Optional[Tuple[str, str]]:
        if self.i >= len(self.left) or self.j >= len(self.right):
            return None
        return self.left[self.i], self.right[self.j]

    def apply_choice(self, winner: str) -> None:
        a, b = self.left[self.i], self.right[self.j]
        if winner == a:
            self.out.append(a)
            self.i += 1
        elif winner == b:
            self.out.append(b)
            self.j += 1
        else:
            raise ValueError("winner must be one of the current pair")

    def is_done(self) -> bool:
        return self.i >= len(self.left) or self.j >= len(self.right)

    def finalize(self) -> List[str]:
        if not self.out:
            self.out = []
        if self.i < len(self.left):
            self.out.extend(self.left[self.i :])
        if self.j < len(self.right):
            self.out.extend(self.right[self.j :])
        return self.out


def init_sort_state(symbols: List[str], *, snapshot_id: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    syms = [str(s) for s in symbols]
    state: Dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "created_at": now_iso_local(),
        "symbols": syms,
        "level": 0,
        "level_runs": [[s] for s in syms],
        "next_runs": [],
        "active_merge": None,
        "comparisons_done": 0,
        "comparisons_est_max": mergesort_estimated_max_comparisons(len(syms)),
        "history": [],  # list of {ts, a, b, choice}
        "phase": "sorting",  # sorting | review | done
        "final_ranking": None,
        "review_queue": [],
        "review_history": [],  # {ts, a, b, choice, previous_choice(optional)}
        "inconsistencies": [],  # {a, b, first_choice, second_choice, ts}
    }
    if meta:
        state["meta"] = meta
    return state


def _start_next_merge_if_needed(state: Dict[str, Any]) -> None:
    if state.get("active_merge") is not None:
        return

    level_runs: List[List[str]] = state.get("level_runs", [])
    next_runs: List[List[str]] = state.get("next_runs", [])

    while True:
        if len(level_runs) >= 2:
            left = level_runs.pop(0)
            right = level_runs.pop(0)
            m = MergeState(left=left, right=right)
            state["active_merge"] = {
                "left": m.left,
                "right": m.right,
                "i": m.i,
                "j": m.j,
                "out": m.out,
            }
            state["level_runs"] = level_runs
            state["next_runs"] = next_runs
            return

        if len(level_runs) == 1:
            next_runs.append(level_runs.pop(0))
            state["level_runs"] = level_runs
            state["next_runs"] = next_runs
            continue

        # level_runs 비었음: 레벨 종료
        if len(next_runs) == 1:
            # 정렬 완료
            state["final_ranking"] = next_runs[0]
            state["phase"] = "review"
            state["active_merge"] = None
            state["level_runs"] = []
            state["next_runs"] = []
            return

        # 다음 레벨 준비
        state["level"] = int(state.get("level", 0)) + 1
        level_runs = next_runs
        next_runs = []
        state["level_runs"] = level_runs
        state["next_runs"] = next_runs
        # 다음 레벨에서 merge 시작 시도


def get_next_comparison(state: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if state.get("phase") != "sorting":
        return None
    _start_next_merge_if_needed(state)
    am = state.get("active_merge")
    if am is None:
        return None
    m = MergeState(left=am["left"], right=am["right"], i=am["i"], j=am["j"], out=am.get("out") or [])
    return m.current_pair()


def apply_sort_choice(state: Dict[str, Any], winner: str, *, ts: Optional[str] = None) -> None:
    if state.get("phase") != "sorting":
        raise ValueError("not in sorting phase")
    _start_next_merge_if_needed(state)
    am = state.get("active_merge")
    if am is None:
        raise ValueError("no active merge")

    m = MergeState(left=am["left"], right=am["right"], i=am["i"], j=am["j"], out=am.get("out") or [])
    pair = m.current_pair()
    if pair is None:
        raise ValueError("merge has no current pair")
    a, b = pair
    m.apply_choice(winner)
    state["comparisons_done"] = int(state.get("comparisons_done", 0)) + 1
    state.setdefault("history", []).append(
        {"ts": ts or now_iso_local(), "a": a, "b": b, "choice": winner}
    )

    if m.is_done():
        merged = m.finalize()
        # active merge 제거
        state["active_merge"] = None
        # 다음 레벨 런에 merged 추가
        nr: List[List[str]] = state.get("next_runs", [])
        nr.append(merged)
        state["next_runs"] = nr
        # 다음 merge 준비
        _start_next_merge_if_needed(state)
    else:
        # 진행 중: 상태 업데이트
        state["active_merge"] = {"left": m.left, "right": m.right, "i": m.i, "j": m.j, "out": m.out}


def build_review_queue_from_final(state: Dict[str, Any], *, fraction: float = 0.10) -> None:
    ranking: Optional[List[str]] = state.get("final_ranking")
    if not ranking:
        return
    n = len(ranking)
    if n < 2:
        state["review_queue"] = []
        return

    import random

    pair_count = max(1, int(round((n - 1) * fraction)))
    indices = list(range(n - 1))
    random.shuffle(indices)
    picked = sorted(indices[:pair_count])
    queue = [(ranking[i], ranking[i + 1]) for i in picked]
    state["review_queue"] = queue


def get_next_review_pair(state: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if state.get("phase") != "review":
        return None
    q = state.get("review_queue") or []
    if not q:
        state["phase"] = "done"
        return None
    return tuple(q[0])  # type: ignore[return-value]


def _find_first_choice(history: List[Dict[str, Any]], a: str, b: str) -> Optional[str]:
    # a/b 순서는 무관
    s = {a, b}
    for h in history:
        if {h.get("a"), h.get("b")} == s:
            return h.get("choice")
    return None


def apply_review_choice(state: Dict[str, Any], winner: str, *, ts: Optional[str] = None) -> Dict[str, Any]:
    if state.get("phase") != "review":
        raise ValueError("not in review phase")
    q = state.get("review_queue") or []
    if not q:
        state["phase"] = "done"
        return {"status": "done"}

    a, b = q.pop(0)
    state["review_queue"] = q

    first_choice = _find_first_choice(state.get("history") or [], a, b)
    review_entry: Dict[str, Any] = {
        "ts": ts or now_iso_local(),
        "a": a,
        "b": b,
        "choice": winner,
    }
    if first_choice is not None:
        review_entry["previous_choice"] = first_choice
        if first_choice != winner:
            inc = {
                "ts": review_entry["ts"],
                "a": a,
                "b": b,
                "first_choice": first_choice,
                "second_choice": winner,
            }
            state.setdefault("inconsistencies", []).append(inc)

            # 인접 쌍이므로 최종 랭킹에 반영(인접 swap)
            ranking: Optional[List[str]] = state.get("final_ranking")
            if ranking:
                try:
                    ia = ranking.index(a)
                    ib = ranking.index(b)
                    if abs(ia - ib) == 1:
                        # a,b가 인접하고 사용자가 b를 선택(즉 b가 더 우수)하면 b가 앞에 오도록 swap
                        if winner == b and ia < ib:
                            ranking[ia], ranking[ib] = ranking[ib], ranking[ia]
                        elif winner == a and ib < ia:
                            ranking[ib], ranking[ia] = ranking[ia], ranking[ib]
                        state["final_ranking"] = ranking
                except ValueError:
                    pass

    state.setdefault("review_history", []).append(review_entry)

    if not state.get("review_queue"):
        state["phase"] = "done"
        return {"status": "done"}
    return {"status": "ok"}

