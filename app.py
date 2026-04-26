# app.py
import io
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.preprocess import preprocess_pil, remove_overlay_text
from utils.inference import load_model, get_anomaly_score
from utils.explain import (
    generate_vit_gradcam,
    compute_quality_metrics,
    enhance_contrast_for_display,
)

PLANE_NAMES = [
    "Plane 1 – Trans-ventricular",
    "Plane 2 – Trans-thalamic",
    "Plane 3 – Trans-cerebellar",
    "Plane 4 – Profile",
    "Plane 5 – Spine",
    "Plane 6 – Abdomen",
]
APP_TITLE = "Fetal Plane AI Viewer"


def quality_badge(metrics):
    bad_count = 0
    for key in ["brightness_label", "contrast_label", "sharpness_label"]:
        if metrics[key] not in ("OK", "Sharp enough"):
            bad_count += 1
    if bad_count == 0:
        return "Good", "#16a34a"
    elif bad_count == 1:
        return "Borderline", "#eab308"
    else:
        return "Poor", "#dc2626"


def plane_comment(plane_name: str, confidence: float) -> str:
    if confidence >= 0.9:
        level = "very high"
    elif confidence >= 0.75:
        level = "high"
    elif confidence >= 0.6:
        level = "moderate"
    else:
        level = "low"
    return (
        f"Predicted **{plane_name}** with {level} confidence "
        f"({confidence*100:.1f}%). Review anatomy and landmarks before deciding."
    )


def explain_prediction(probs, plane_names):
    idx = np.argsort(-probs)[:2]
    p1, p2 = idx[0], idx[1]
    c1, c2 = probs[p1], probs[p2]
    text = (
        f"Model is primarily between **{plane_names[p1]}** ({c1*100:.1f}%) "
        f"and **{plane_names[p2]}** ({c2*100:.1f}%)."
    )
    if c1 - c2 < 0.15:
        text += " The two planes are relatively close; treat this as a softer decision."
    else:
        text += " The top class is clearly preferred."
    return text


def predict_one(model, device, pil_img: Image.Image):
    x = preprocess_pil(pil_img, device, clean_overlay=True)

    with torch.no_grad():
        feats, logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_plane = PLANE_NAMES[pred_idx]

    feat_vec = feats.cpu().numpy().reshape(-1)
    raw_anom, anom_pct = get_anomaly_score(feat_vec)

    return pred_plane, probs, pred_idx, raw_anom, anom_pct


def anomaly_label(anom_pct: float) -> str:
    if anom_pct < 20:
        return "Very typical", "#16a34a"
    if anom_pct < 40:
        return "Mostly typical", "#22c55e"
    if anom_pct < 70:
        return "Borderline atypical", "#eab308"
    return "Potential anomaly", "#dc2626"


def process_image(model, device, f, show_gradcam, enhanced_first, show_probs):
    bytes_data = f.read()
    pil_img = Image.open(io.BytesIO(bytes_data))

    # Container
    st.markdown(
        "<div style='margin:16px auto; padding:20px; border-radius:12px; "
        "border:1px solid #2d3748; background:#1a202c; max-width:1400px;'>",
        unsafe_allow_html=True,
    )

    # Header row
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {f.name}")
    with col_badge:
        metrics_for_badge = compute_quality_metrics(remove_overlay_text(pil_img))
        q_label, q_color = quality_badge(metrics_for_badge)
        st.markdown(
            f"<div style='text-align:right;'>"
            f"<span style='padding:4px 12px; border-radius:20px; "
            f"background:{q_color}; color:white; font-size:12px; font-weight:600;'>"
            f"Quality: {q_label}</span></div>",
            unsafe_allow_html=True,
        )

    col_main, col_side = st.columns([1.5, 1])

    # Tabs
    if enhanced_first:
        tab_names = ["Enhanced view", "Original", "Heatmap"]
    else:
        tab_names = ["Original", "Heatmap", "Enhanced view"]

    with col_main:
        tabs = st.tabs(tab_names)

        def idx(name): return tab_names.index(name)

        with tabs[idx("Original")]:
            st.image(pil_img, use_column_width=True)

        with tabs[idx("Heatmap")]:
            if show_gradcam:
                with st.spinner("Computing Grad‑CAM..."):
                    clean_for_cam = remove_overlay_text(pil_img)
                    heatmap_img = generate_vit_gradcam(
                        model, device, clean_for_cam, target_class_idx=None
                    )
                st.image(heatmap_img, use_column_width=True)
            else:
                st.info("Enable Grad‑CAM in the sidebar.")

        with tabs[idx("Enhanced view")]:
            cleaned = remove_overlay_text(pil_img)
            enhanced = enhance_contrast_for_display(cleaned)
            st.image(enhanced, use_column_width=True)

            with st.expander("📊 Image quality details"):
                metrics = compute_quality_metrics(cleaned)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Brightness", f"{metrics['brightness']:.3f}", metrics['brightness_label'])
                with col2:
                    st.metric("Contrast", f"{metrics['contrast']:.3f}", metrics['contrast_label'])
                with col3:
                    st.metric("Sharpness", f"{metrics['sharpness']:.4f}", metrics['sharpness_label'])

    with col_side:
        with st.spinner("Running model..."):
            pred_plane, probs, pred_idx, raw_anom, anom_pct = predict_one(
                model, device, pil_img
            )

        top_conf = float(probs[pred_idx])

        # Prediction card
        st.markdown(
            "<div style='padding:16px; border-radius:10px; "
            "background:linear-gradient(135deg,#10b981,#059669); "
            "color:white; margin-bottom:12px; text-align:center;'>"
            "<div style='font-size:13px; opacity:0.9;'>Predicted Plane</div>"
            f"<div style='font-size:20px; font-weight:700; margin:8px 0;'>{pred_plane}</div>"
            f"<div style='font-size:14px;'>Confidence: {top_conf*100:.1f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Anomaly
        anom_text, anom_color = anomaly_label(anom_pct)
        st.markdown(
            f"<div style='padding:12px; border-radius:8px; background:{anom_color}20; "
            f"border:1px solid {anom_color}; margin-bottom:12px;'>"
            f"<div style='color:{anom_color}; font-weight:600; font-size:13px;'>Anomaly Score</div>"
            f"<div style='color:{anom_color}; font-size:16px; font-weight:700;'>{anom_text} ({anom_pct:.1f}%)</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Summary
        st.markdown("#### Summary")
        st.write(plane_comment(pred_plane, top_conf))
        st.caption(explain_prediction(probs, PLANE_NAMES))

        # Probabilities
        st.markdown("#### Plane Probabilities")
        prob_percent = [float(p * 100.0) for p in probs]

        if show_probs:
            # Plotly horizontal bar
            fig = go.Figure(go.Bar(
                x=prob_percent,
                y=PLANE_NAMES,
                orientation='h',
                marker=dict(
                    color=prob_percent,
                    colorscale='Teal',
                    line=dict(color='#1a202c', width=1)
                ),
                text=[f"{p:.1f}%" for p in prob_percent],
                textposition='outside',
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title="Probability (%)",
                    gridcolor='#2d3748',
                    range=[0, 100]
                ),
                yaxis=dict(
                    title="",
                    gridcolor='#2d3748'
                ),
                font=dict(size=11, color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            df = pd.DataFrame({
                "Plane": PLANE_NAMES,
                "Probability (%)": [round(p, 2) for p in prob_percent],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "image_name": f.name,
        "predicted_plane": pred_plane,
        "confidence(%)": round(top_conf * 100.0, 2),
        "anomaly_score_raw": round(raw_anom, 4),
        "anomaly_percentage(%)": round(anom_pct, 2),
        **{
            f"{name}_prob(%)": round(p * 100.0, 2)
            for name, p in zip(PLANE_NAMES, probs)
        },
    }


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.markdown("# ⚙️ Controls")
        st.markdown("---")

        st.markdown("### 📁 Mode")
        compare_mode = st.checkbox("Two-study compare", value=False, help="Compare previous vs current study")
        batch_mode = st.checkbox("Multiple images", value=True, help="Upload and analyze multiple frames")
        enhanced_first = st.checkbox("Enhanced tab first", value=True, help="Show enhanced view as default tab")

        st.markdown("### 🎨 Visualization")
        show_probs = st.checkbox("Probability chart", value=True, help="Show interactive probability chart")
        show_gradcam = st.checkbox("Grad-CAM heatmap", value=True, help="Show model attention heatmap")

        st.markdown("### 🔍 Filters")
        plane_filter = st.selectbox(
            "Filter by plane",
            ["All"] + PLANE_NAMES,
            help="Filter case summary by predicted plane"
        )
        anom_min = st.slider(
            "Min anomaly %",
            0, 100, 0,
            help="Show only images with anomaly score above this threshold"
        )

        st.markdown("---")
        st.caption("⚠️ AI predictions only. Use as decision support, not standalone diagnosis.")

    # Header
    st.markdown(
        "<div style='padding:20px; border-radius:12px; "
        "background:linear-gradient(90deg,#1a202c,#2d3748); "
        "color:#e2e8f0; margin-bottom:20px; text-align:center;'>"
        f"<h1 style='margin:0; font-size:28px; color:#10b981;'>{APP_TITLE}</h1>"
        "<p style='margin:8px 0 0 0; font-size:14px; color:#cbd5e0;'>"
        "Advanced AI-powered fetal plane classification with explainability"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Model
    model_device = load_model()
    if model_device is None:
        st.stop()
    model, device = model_device

    # Upload
    st.markdown("## 📤 Upload")

    if compare_mode:
        col1, col2 = st.columns(2)
        with col1:
            prev_files = st.file_uploader(
                "📁 Previous study",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="prev",
            )
        with col2:
            curr_files = st.file_uploader(
                "📁 Current study",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="curr",
            )

        if not prev_files and not curr_files:
            st.info("↑ Upload at least one image from each study to compare.")
            return

        st.markdown("---")
        st.markdown("## 🔬 Two-Study Comparison")

        results_prev = []
        results_curr = []

        for i, (pf, cf) in enumerate(zip(prev_files or [], curr_files or []), start=1):
            st.markdown(f"### Pair {i}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Previous Study**")
                res = process_image(model, device, pf, show_gradcam, enhanced_first, show_probs)
                results_prev.append(res)
            with col2:
                st.markdown("**Current Study**")
                res = process_image(model, device, cf, show_gradcam, enhanced_first, show_probs)
                results_curr.append(res)

        results = results_prev + results_curr

    else:
        uploaded_files = st.file_uploader(
            "Choose ultrasound images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=batch_mode,
        )

        if not uploaded_files:
            st.info("↑ Upload at least one ultrasound image to begin analysis.")
            return

        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]

        results = []

        st.markdown("---")
        st.markdown("## 🔬 Analysis")

        for f in uploaded_files:
            res = process_image(model, device, f, show_gradcam, enhanced_first, show_probs)
            results.append(res)

    # Summary
    st.markdown("---")
    st.markdown("## 📊 Case Summary")

    report_df = pd.DataFrame(results)

    # Filter
    filtered = report_df.copy()
    if plane_filter != "All":
        filtered = filtered[filtered["predicted_plane"] == plane_filter]
    filtered = filtered[filtered["anomaly_percentage(%)"] >= anom_min]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "image_name": st.column_config.TextColumn("Image", width="medium"),
            "predicted_plane": st.column_config.TextColumn("Predicted Plane", width="large"),
            "confidence(%)": st.column_config.NumberColumn("Confidence (%)", format="%.2f"),
            "anomaly_percentage(%)": st.column_config.NumberColumn("Anomaly (%)", format="%.2f"),
        }
    )

    csv_bytes = report_df.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "📥 Download CSV Report",
        data=csv_bytes,
        file_name=f"fetal_plane_report_{ts}.csv",
        mime="text/csv",
    )

    # Analytics
    if len(report_df) > 1:
        st.markdown("---")
        st.markdown("## 📈 Session Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Plane Distribution")
            plane_counts = report_df["predicted_plane"].value_counts()
            fig = px.pie(
                values=plane_counts.values,
                names=plane_counts.index,
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Confidence Distribution")
            fig = go.Figure(go.Histogram(
                x=report_df["confidence(%)"],
                nbinsx=10,
                marker=dict(color='#10b981', line=dict(color='#1a202c', width=1))
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Confidence (%)", gridcolor='#2d3748'),
                yaxis=dict(title="Count", gridcolor='#2d3748'),
                font=dict(size=11, color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("#### Anomaly Distribution")
            fig = go.Figure(go.Histogram(
                x=report_df["anomaly_percentage(%)"],
                nbinsx=10,
                marker=dict(color='#eab308', line=dict(color='#1a202c', width=1))
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Anomaly (%)", gridcolor='#2d3748'),
                yaxis=dict(title="Count", gridcolor='#2d3748'),
                font=dict(size=11, color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: confidence vs anomaly
        st.markdown("#### Confidence vs Anomaly")
        fig = px.scatter(
            report_df,
            x="confidence(%)",
            y="anomaly_percentage(%)",
            color="predicted_plane",
            hover_data=["image_name"],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Confidence (%)", gridcolor='#2d3748'),
            yaxis=dict(title="Anomaly (%)", gridcolor='#2d3748'),
            font=dict(size=12, color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
