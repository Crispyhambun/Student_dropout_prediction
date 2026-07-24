import sys
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_analytics_dashboard_renders_all_plots():
    at = AppTest.from_file("pages/3_Analytics_Dashboard.py")
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "📊 Analytics Dashboard"

    subheaders = [s.value for s in at.subheader]
    assert "Feature Importance (Random Forest)" in subheaders
    assert "Age vs GPA (Semester 2 Grade)" in subheaders
    assert "Correlation Heatmap" in subheaders

    # Dropout-by-course, risk distribution, feature importance, age/GPA scatter,
    # and the correlation heatmap should all render since model + dataset exist.
    assert len(at.get("plotly_chart")) == 5


def test_analytics_dashboard_shows_warning_when_model_missing():
    at = AppTest.from_file("pages/3_Analytics_Dashboard.py")
    with mock.patch(
        "utils.preprocess.load_model_artifacts", side_effect=FileNotFoundError("no model")
    ):
        at.run(timeout=30)

    assert not at.exception
    assert any("Could not load model artifacts" in w.value for w in at.warning)
    assert any("view feature importance" in i.value for i in at.info)


def test_analytics_dashboard_shows_error_when_dataset_missing():
    at = AppTest.from_file("pages/3_Analytics_Dashboard.py")
    with mock.patch("pathlib.Path.exists", return_value=False):
        at.run(timeout=30)

    assert not at.exception
    assert any("Dataset not found" in e.value for e in at.error)
