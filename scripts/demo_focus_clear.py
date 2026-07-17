# -*- coding: utf-8 -*-
"""세부보기 selectbox 포커스 시 입력 초기화 CSS/JS 검증용 데모 (수동 실행 전용).

    streamlit run scripts/demo_focus_clear.py --server.port 8599
"""

import streamlit as st
import streamlit.components.v1 as components

st.title("focus-clear demo")

options = [f"[{i:02d}] TICKER{i} (Name {i})" for i in range(1, 31)]

st.selectbox(
    "종목 선택",
    options=options,
    index=4,
    key="detail_selectbox_4",
    label_visibility="collapsed",
)

st.selectbox("다른 selectbox (영향 없어야 함)", options=options, index=2, key="other_box")

# 포커스 시 선택 라벨 숨김 + 주입용 iframe 요소의 레이아웃 공백 제거
st.markdown(
    """
    <style>
    [class*="st-key-detail_selectbox"] [data-baseweb="select"]:has(input[aria-expanded="true"]) div[value] {
        display: none;
    }
    div[data-testid="stElementContainer"]:has(iframe[height="0"]) {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (function () {
        const doc = window.parent.document;
        if (doc.__detailSelectboxFocusClear) { return; }
        doc.__detailSelectboxFocusClear = true;
        const inDetailBox = function (el) {
            return el && el.tagName === 'INPUT'
                && el.closest('[class*="st-key-detail_selectbox"]');
        };
        const setter = Object.getOwnPropertyDescriptor(
            window.parent.HTMLInputElement.prototype, 'value').set;
        doc.addEventListener('focusin', function (ev) {
            const input = ev.target;
            if (!inDetailBox(input)) { return; }
            window.setTimeout(function () {
                if (doc.activeElement !== input || !input.value) { return; }
                setter.call(input, '');
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }, 0);
        }, true);
        doc.addEventListener('keydown', function (ev) {
            if (ev.key !== 'Backspace') { return; }
            const input = ev.target;
            if (!inDetailBox(input)) { return; }
            if (!input.value) {
                ev.preventDefault();
                ev.stopPropagation();
            }
        }, true);
    })();
    </script>
    """,
    height=0,
)
