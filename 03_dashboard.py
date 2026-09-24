"""
SIEM AI - Kill Chain Dashboard (03)
Streamlit app hiển thị attack session, Kill Chain timeline, risk score.
Nguồn dữ liệu: output/attack_sessions.json (sinh bởi 02_correlation_engine.py)

Chạy: streamlit run 03_dashboard.py
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def _load_sessions(path):
    """Đọc attack_sessions.json 1 lần, cache lại giữa các lần Streamlit rerun."""
    with open(path, encoding="utf-8") as f:
        sessions = json.load(f)
    df = pd.DataFrame(sessions)
    df["phases_pred_str"] = df["phases_pred"].apply(lambda x: " → ".join(x) if x else "—")
    df["phases_gt_str"] = df["phases_gt"].apply(lambda x: " → ".join(x) if x else "—")
    df["is_attack_pred"] = df["phases_pred"].apply(len) > 0
    df["is_attack_gt"] = df["phases_gt"].apply(len) > 0
    df["start_dt"] = pd.to_datetime(df["start_time"], unit="s")
    df["risk_level"] = pd.Categorical(
        df["risk_level"], categories=KillChainDashboardApp.RISK_ORDER, ordered=True
    )
    return sessions, df


class KillChainDashboardApp:
    """Dashboard Streamlit: bộ lọc + KPI + chart + bảng session + timeline chi tiết."""

    OUTPUT_DIR = "output"
    SESSIONS_PATH = f"{OUTPUT_DIR}/attack_sessions.json"
    TEST_SCENARIOS = ["fox", "harrison"]

    PHASE_ORDER = ["Reconnaissance", "Exploitation", "Installation", "Actions_on_Objectives"]
    RISK_ORDER = ["BENIGN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # Màu — lấy từ palette CVD-safe đã validate (categorical slot 1-4 theo thứ tự
    # Kill Chain cố định; status palette cho risk_level)
    PHASE_COLOR = {
        "Reconnaissance":        "#2a78d6",  # slot 1 blue
        "Exploitation":          "#eb6834",  # slot 2 orange
        "Installation":          "#1baf7a",  # slot 3 aqua
        "Actions_on_Objectives": "#eda100",  # slot 4 yellow
    }
    RISK_COLOR = {
        "BENIGN":   "#898781",  # muted gray — không phải status, chỉ "không rủi ro"
        "LOW":      "#0ca30c",  # good
        "MEDIUM":   "#fab219",  # warning
        "HIGH":     "#ec835a",  # serious
        "CRITICAL": "#d03b3b",  # critical
    }
    INK_SECONDARY = "#52514e"
    GRID_COLOR = "#e1e0d9"

    def __init__(self):
        st.set_page_config(page_title="SIEM AI — Kill Chain Dashboard", page_icon="🛡️", layout="wide")
        try:
            self.sessions_raw, self.df_all = _load_sessions(self.SESSIONS_PATH)
        except FileNotFoundError:
            st.error(f"Không tìm thấy `{self.SESSIONS_PATH}`. Chạy `python3 02_correlation_engine.py` trước.")
            st.stop()
        self.sessions_by_id = {s["session_id"]: s for s in self.sessions_raw}
        self.df = self.df_all  # bị thu hẹp bởi render_sidebar()

    # ── SIDEBAR — FILTERS ────────────────────────────────────────────
    def render_sidebar(self):
        st.sidebar.header("Bộ lọc")

        scenarios = sorted(self.df_all["scenario"].unique().tolist())
        self.scenarios = scenarios
        sel_scenarios = st.sidebar.multiselect("Scenario", scenarios, default=scenarios)
        sel_risk = st.sidebar.multiselect("Risk level", self.RISK_ORDER, default=self.RISK_ORDER)
        only_multi = st.sidebar.checkbox("Chỉ hiện session multi-phase", value=False)
        search_ip = st.sidebar.text_input("Tìm theo source IP (chứa)", "")

        df = self.df_all[
            self.df_all["scenario"].isin(sel_scenarios) &
            self.df_all["risk_level"].isin(sel_risk)
        ]
        if only_multi:
            df = df[df["is_multi_phase"]]
        if search_ip.strip():
            df = df[df["source_ip"].str.contains(search_ip.strip(), na=False)]

        self.df = df
        st.sidebar.caption(f"{len(df):,} / {len(self.df_all):,} session sau lọc")

    # ── HEADER + KPI ──────────────────────────────────────────────────
    def render_header_and_kpis(self):
        st.title("🛡️ SIEM AI — Kill Chain Dashboard")
        st.caption("Attack session được dựng từ alert bằng Correlation Engine (RF v2 + threshold tuning)")

        df = self.df
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Tổng session", f"{len(df):,}")
        k2.metric("Session bị gắn attack", f"{int(df['is_attack_pred'].sum()):,}")
        k3.metric("CRITICAL", f"{int((df['risk_level'] == 'CRITICAL').sum()):,}")
        k4.metric("HIGH", f"{int((df['risk_level'] == 'HIGH').sum()):,}")
        k5.metric("Multi-phase", f"{int(df['is_multi_phase'].sum()):,}")
        st.divider()

    # ── CHARTS ────────────────────────────────────────────────────────
    def _bar_layout(self, fig, yaxis_title, showlegend=False):
        fig.update_layout(
            showlegend=showlegend, xaxis_title=None, yaxis_title=yaxis_title,
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
            font_color=self.INK_SECONDARY, yaxis_gridcolor=self.GRID_COLOR,
            margin=dict(t=10, b=10),
        )
        return fig

    def render_charts(self):
        df = self.df
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Phân bố Risk Level")
            risk_counts = (
                df["risk_level"].value_counts().reindex(self.RISK_ORDER, fill_value=0).reset_index()
            )
            risk_counts.columns = ["risk_level", "count"]
            fig = px.bar(risk_counts, x="risk_level", y="count",
                         color="risk_level", color_discrete_map=self.RISK_COLOR, text="count")
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(self._bar_layout(fig, "Số session"), width='stretch')

        with c2:
            st.subheader("Phase cao nhất đạt được (predicted)")
            phase_counts = (
                df[df["max_phase_pred"].notna()]["max_phase_pred"]
                .value_counts().reindex(self.PHASE_ORDER, fill_value=0).reset_index()
            )
            phase_counts.columns = ["phase", "count"]
            fig = px.bar(phase_counts, x="phase", y="count",
                         color="phase", color_discrete_map=self.PHASE_COLOR, text="count")
            fig.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(self._bar_layout(fig, "Số session"), width='stretch')

        st.subheader("Session theo Scenario, chia theo Risk Level")
        scen_risk = df.groupby(["scenario", "risk_level"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            scen_risk, x="scenario", y="count", color="risk_level",
            color_discrete_map=self.RISK_COLOR,
            category_orders={"risk_level": self.RISK_ORDER, "scenario": self.scenarios},
        )
        fig.update_traces(marker_line_width=0, marker_line_color="#fcfcfb")
        fig.update_layout(
            barmode="stack", xaxis_title=None, yaxis_title="Số session",
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
            font_color=self.INK_SECONDARY, yaxis_gridcolor=self.GRID_COLOR,
            legend_title_text="Risk level", margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')
        st.divider()

    # ── SESSION TABLE ─────────────────────────────────────────────────
    def render_table(self):
        st.subheader("Danh sách Attack Session")
        table_cols = [
            "session_id", "scenario", "source_ip", "start_dt", "duration_min",
            "alert_count", "phases_pred_str", "risk_score", "risk_level",
            "status", "recommended_action",
        ]
        df_table = self.df.sort_values("risk_score", ascending=False)[table_cols].rename(columns={
            "session_id": "Session ID", "scenario": "Scenario", "source_ip": "Source IP",
            "start_dt": "Bắt đầu", "duration_min": "Thời lượng (phút)",
            "alert_count": "Số alert", "phases_pred_str": "Kill Chain (pred)",
            "risk_score": "Risk score", "risk_level": "Risk level",
            "status": "Trạng thái", "recommended_action": "Hành động đề xuất",
        })
        st.dataframe(df_table, width='stretch', height=380, hide_index=True)
        st.divider()

    # ── SESSION DETAIL — KILL CHAIN TIMELINE ─────────────────────────
    def _render_timeline_chart(self, sess):
        kc = sess["kill_chain_progress"]
        if not kc:
            st.info("Session này không có phase attack nào được model phát hiện.")
            return

        t0 = sess["start_time"]
        timeline_df = pd.DataFrame(kc)
        timeline_df["t_min"] = (timeline_df["first_seen"] - t0) / 60
        timeline_df["y"] = timeline_df["phase"].apply(
            lambda p: self.PHASE_ORDER.index(p) if p in self.PHASE_ORDER else -1
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_df["t_min"], y=timeline_df["y"], mode="lines",
            line=dict(color=self.GRID_COLOR, width=2), hoverinfo="skip", showlegend=False,
        ))
        for phase in self.PHASE_ORDER:
            sub = timeline_df[timeline_df["phase"] == phase]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["t_min"], y=sub["y"], mode="markers",
                marker=dict(size=18, color=self.PHASE_COLOR[phase], line=dict(width=2, color="#fcfcfb")),
                name=phase,
                customdata=sub[["phase", "alert_count", "top_alert"]],
                hovertemplate="<b>%{customdata[0]}</b><br>t=+%{x:.1f} phút<br>"
                              "%{customdata[1]} alert (top: %{customdata[2]})<extra></extra>",
            ))
        fig.update_layout(
            yaxis=dict(tickmode="array", tickvals=list(range(len(self.PHASE_ORDER))),
                       ticktext=self.PHASE_ORDER, range=[-0.5, len(self.PHASE_ORDER) - 0.5],
                       gridcolor=self.GRID_COLOR),
            xaxis=dict(title="Phút kể từ khi session bắt đầu", gridcolor=self.GRID_COLOR),
            plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
            font_color=self.INK_SECONDARY, height=320,
            legend_title_text="Phase", margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig, width='stretch')

    def render_detail(self):
        st.subheader("Chi tiết Session — Kill Chain Timeline")

        if len(self.df) == 0:
            st.info("Không có session nào khớp bộ lọc.")
            st.divider()
            return

        options = self.df.sort_values("risk_score", ascending=False)["session_id"].tolist()
        sel_id = st.selectbox("Chọn session", options, index=0)
        sess = self.sessions_by_id[sel_id]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Source IP", sess["source_ip"])
        m2.metric("Risk score", sess["risk_score"], help=sess["risk_level"])
        m3.metric("Số alert", f"{sess['alert_count']:,}")
        m4.metric("Thời lượng", f"{sess['duration_min']:.0f} phút")

        badge_color = self.RISK_COLOR.get(sess["risk_level"], "#898781")
        st.markdown(
            f"<span style='background:{badge_color};color:white;padding:4px 10px;"
            f"border-radius:4px;font-weight:600'>{sess['risk_level']}</span> "
            f"&nbsp; **{sess['recommended_action']}**",
            unsafe_allow_html=True,
        )

        self._render_timeline_chart(sess)

        gt_str = " → ".join(sess["phases_gt"]) if sess["phases_gt"] else "Benign only"
        pred_str = " → ".join(sess["phases_pred"]) if sess["phases_pred"] else "Benign only"
        cc1, cc2 = st.columns(2)
        cc1.markdown(f"**Ground truth:** {gt_str}")
        cc2.markdown(f"**Predicted:** {pred_str}")
        if set(sess["phases_gt"]) != set(sess["phases_pred"]):
            st.warning("⚠️ Predicted khác Ground truth cho session này.")

        with st.expander("JSON đầy đủ"):
            st.json(sess)

        st.divider()

    # ── MODEL EVALUATION (tính động trên phạm vi được chọn) ──────────
    def render_evaluation(self):
        st.subheader("Đánh giá Correlation Engine")

        eval_scope = st.radio(
            "Phạm vi đánh giá",
            ["Test scenarios (fox, harrison)", "Tất cả 8 scenario", "Theo bộ lọc hiện tại"],
            horizontal=True,
        )
        if eval_scope.startswith("Test"):
            df_eval = self.df_all[self.df_all["scenario"].isin(self.TEST_SCENARIOS)]
        elif eval_scope.startswith("Tất cả"):
            df_eval = self.df_all
        else:
            df_eval = self.df

        tp = int((df_eval["is_attack_pred"] & df_eval["is_attack_gt"]).sum())
        fp = int((df_eval["is_attack_pred"] & ~df_eval["is_attack_gt"]).sum())
        fn = int((~df_eval["is_attack_pred"] & df_eval["is_attack_gt"]).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision", f"{precision:.3f}")
        e2.metric("Recall", f"{recall:.3f}")
        e3.metric("F1-score", f"{f1:.3f}")
        e4.metric("TP / FP / FN", f"{tp} / {fp} / {fn}")

        st.info(
            "**Đã biết:** phần lớn false positive còn lại là session bị gắn nhầm "
            "phase Reconnaissance, do 2 alert code `W-All-Evt` / `W-Acc-Cms` mà model "
            "học sai association ở train scenarios — giới hạn generalization của model "
            "gốc, không phải lỗi threshold. Xem `CLAUDE.md` mục 'Vấn đề còn tồn tại'."
        )

    def run(self):
        self.render_sidebar()
        self.render_header_and_kpis()
        self.render_charts()
        self.render_table()
        self.render_detail()
        self.render_evaluation()


if __name__ == "__main__":
    KillChainDashboardApp().run()
