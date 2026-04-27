from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase


class ClassPriorPlot(QWidget):
    """Bar chart for Naive Bayes class priors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._data: list[tuple[str, float]] = []

    def set_data(self, priors: list[tuple[str, float]]) -> None:
        self._data = priors
        self.update()

    def paintEvent(self, event) -> None:
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - margin * 2
        plot_h = h - margin * 2

        bar_w = plot_w / len(self._data)
        max_p = max(p for _, p in self._data) if self._data else 1.0

        painter.setFont(QFont("Inter", 8))

        colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

        for i, (label, prob) in enumerate(self._data):
            bar_h = (prob / max_p) * plot_h
            x = margin + i * bar_w + 5
            y = h - margin - bar_h

            color = QColor(colors[i % len(colors)])
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_w - 10, bar_h), 4, 4)

            painter.setPen(QColor("#475569"))
            painter.drawText(QRectF(x, h - margin + 5, bar_w - 10, 20), Qt.AlignmentFlag.AlignCenter, label)
            painter.drawText(QRectF(x, y - 20, bar_w - 10, 20), Qt.AlignmentFlag.AlignCenter, f"{prob:.2f}")


class NaiveBayesScreen(ModelScreenBase):
    """Gaussian Naive Bayes — fast probabilistic classifier."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # Side-by-side layout
        h_layout = QHBoxLayout()

        # Left Column: Priors and Info
        left_col = QVBoxLayout()
        
        info_box = QGroupBox(i18n.t("Naive Bayes Analysis"))
        info_layout = QVBoxLayout(info_box)
        lbl = QLabel(
            "Naive Bayes with the assumption of feature independence. "
            "Supports only classification tasks. No configurable parameters."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        info_layout.addWidget(lbl)
        left_col.addWidget(info_box)

        priors_box = QGroupBox(i18n.t("Class Priors"))
        priors_layout = QVBoxLayout(priors_box)
        self._priors_plot = ClassPriorPlot()
        priors_layout.addWidget(self._priors_plot)
        left_col.addWidget(priors_box)
        
        h_layout.addLayout(left_col, 1)

        # Right Column: Theta Table
        theta_box = QGroupBox(i18n.t("Feature Means (Theta)"))
        theta_layout = QVBoxLayout(theta_box)
        self._theta_table = QTableWidget()
        self._theta_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._theta_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._theta_table.setStyleSheet("""
            QTableWidget { border: none; background: transparent; }
            QHeaderView::section { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 4px; font-weight: bold; }
        """)
        theta_layout.addWidget(self._theta_table)
        h_layout.addWidget(theta_box, 2)

        layout.addLayout(h_layout)

    def _train(self):
        from sklearn.naive_bayes import GaussianNB

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"categorical", "boolean"}:
            raise ValueError("Naive Bayes supports only classification targets.")
        
        result = self._svc.fit(GaussianNB(), ds, "Naive Bayes", "naive_bayes", {})
        
        # Extract GaussianNB specific info
        model = result.trained_model
        if hasattr(model, "class_prior_") and hasattr(model, "theta_"):
            # Update Priors
            classes = result.class_values or []
            priors = list(zip(classes, model.class_prior_))
            self._priors_plot.set_data(priors)
            
            # Update Theta Table
            features = result.feature_names
            self._theta_table.setRowCount(len(features))
            self._theta_table.setColumnCount(len(classes))
            self._theta_table.setHorizontalHeaderLabels(classes)
            self._theta_table.setVerticalHeaderLabels(features)
            
            for i, f_name in enumerate(features):
                for j, c_name in enumerate(classes):
                    val = model.theta_[j, i]
                    item = QTableWidgetItem(f"{val:.4f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._theta_table.setItem(i, j, item)
                    
        return result
