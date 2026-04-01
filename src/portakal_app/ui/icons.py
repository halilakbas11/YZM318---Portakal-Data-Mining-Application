from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QStyle


ICON_MAP = {
    "file": QStyle.StandardPixmap.SP_FileIcon,
    "csv": QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "dataset": QStyle.StandardPixmap.SP_DirIcon,
    "table": QStyle.StandardPixmap.SP_FileDialogListView,
    "paint": QStyle.StandardPixmap.SP_BrowserReload,
    "info": QStyle.StandardPixmap.SP_MessageBoxInformation,
    "rank": QStyle.StandardPixmap.SP_ArrowUp,
    "edit": QStyle.StandardPixmap.SP_FileDialogContentsView,
    "color": QStyle.StandardPixmap.SP_DriveDVDIcon,
    "stats": QStyle.StandardPixmap.SP_ComputerIcon,
    "save": QStyle.StandardPixmap.SP_DialogSaveButton,
    "columns": QStyle.StandardPixmap.SP_TitleBarShadeButton,
    "normalize": QStyle.StandardPixmap.SP_ArrowRight,
    "scatter": QStyle.StandardPixmap.SP_DialogYesButton,
    "model": QStyle.StandardPixmap.SP_CommandLink,
    "score": QStyle.StandardPixmap.SP_DialogApplyButton,
    "pca": QStyle.StandardPixmap.SP_ArrowDown,
}


def _make_custom_icon(icon_name: str) -> QIcon | None:
    """Draw a custom 24x24 icon for transform widgets using QPainter."""
    size = 24
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#e07020"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    accent = QColor("#e07020")

    drawn = True

    if icon_name == "index":
        # List with index numbers: 1. 2. 3.
        painter.drawLine(QPointF(5, 6), QPointF(19, 6))
        painter.drawLine(QPointF(5, 12), QPointF(19, 12))
        painter.drawLine(QPointF(5, 18), QPointF(19, 18))
        painter.setPen(QPen(accent, 1.5))
        font = QFont("Consolas", 6)
        painter.setFont(font)
        painter.drawText(QRectF(1, 2, 6, 8), Qt.AlignmentFlag.AlignCenter, "1")
        painter.drawText(QRectF(1, 8, 6, 8), Qt.AlignmentFlag.AlignCenter, "2")
        painter.drawText(QRectF(1, 14, 6, 8), Qt.AlignmentFlag.AlignCenter, "3")

    elif icon_name == "randomize":
        # Shuffle arrows (crossing)
        painter.drawLine(QPointF(4, 7), QPointF(14, 17))
        painter.drawLine(QPointF(4, 17), QPointF(14, 7))
        painter.drawLine(QPointF(14, 7), QPointF(11, 5))
        painter.drawLine(QPointF(14, 7), QPointF(11, 9))
        painter.drawLine(QPointF(14, 17), QPointF(11, 15))
        painter.drawLine(QPointF(14, 17), QPointF(11, 19))

    elif icon_name == "purge":
        # Trash can
        painter.drawRect(QRectF(6, 8, 12, 13))
        painter.drawLine(QPointF(4, 8), QPointF(20, 8))
        painter.drawLine(QPointF(9, 5), QPointF(15, 5))
        painter.drawLine(QPointF(9, 5), QPointF(9, 8))
        painter.drawLine(QPointF(15, 5), QPointF(15, 8))
        painter.drawLine(QPointF(10, 11), QPointF(10, 18))
        painter.drawLine(QPointF(14, 11), QPointF(14, 18))

    elif icon_name == "unique":
        # Fingerprint / checkmark with unique dot
        painter.drawLine(QPointF(5, 13), QPointF(9, 17))
        painter.drawLine(QPointF(9, 17), QPointF(19, 7))
        painter.setPen(QPen(accent, 3.0))
        painter.drawPoint(QPointF(18, 17))

    elif icon_name == "domain":
        # Two overlapping rectangles (template apply)
        painter.drawRect(QRectF(3, 3, 12, 10))
        painter.setPen(QPen(accent.darker(120), 2.0, Qt.PenStyle.DashLine))
        painter.drawRect(QRectF(9, 11, 12, 10))

    elif icon_name == "sampler":
        # Pie chart slice
        painter.drawEllipse(QRectF(3, 3, 18, 18))
        painter.drawLine(QPointF(12, 12), QPointF(12, 3))
        painter.drawLine(QPointF(12, 12), QPointF(20, 8))
        painter.setBrush(QColor("#e07020"))
        path = QPainterPath()
        path.moveTo(12, 12)
        path.arcTo(QRectF(3, 3, 18, 18), 90, -55)
        path.closeSubpath()
        painter.drawPath(path)

    elif icon_name == "columns":
        # Three column bars
        painter.setBrush(accent)
        painter.drawRect(QRectF(3, 4, 4, 16))
        painter.setBrush(accent.lighter(130))
        painter.drawRect(QRectF(10, 4, 4, 16))
        painter.setBrush(accent.lighter(160))
        painter.drawRect(QRectF(17, 4, 4, 16))

    elif icon_name == "rows":
        # Horizontal filter lines with funnel
        painter.drawLine(QPointF(4, 6), QPointF(20, 6))
        painter.drawLine(QPointF(6, 11), QPointF(18, 11))
        painter.drawLine(QPointF(8, 16), QPointF(16, 16))
        # Funnel hint
        painter.drawLine(QPointF(4, 6), QPointF(8, 16))
        painter.drawLine(QPointF(20, 6), QPointF(16, 16))

    elif icon_name == "transpose":
        # 90-degree rotation arrow
        painter.drawArc(QRectF(4, 4, 16, 16), 0, 270 * 16)
        painter.drawLine(QPointF(12, 4), QPointF(9, 7))
        painter.drawLine(QPointF(12, 4), QPointF(15, 7))

    elif icon_name == "split":
        # One line splitting into two
        painter.drawLine(QPointF(4, 12), QPointF(12, 12))
        painter.drawLine(QPointF(12, 12), QPointF(20, 6))
        painter.drawLine(QPointF(12, 12), QPointF(20, 18))

    elif icon_name == "merge":
        # Two lines merging into one
        painter.drawLine(QPointF(4, 6), QPointF(12, 12))
        painter.drawLine(QPointF(4, 18), QPointF(12, 12))
        painter.drawLine(QPointF(12, 12), QPointF(20, 12))
        painter.drawLine(QPointF(17, 9), QPointF(20, 12))
        painter.drawLine(QPointF(17, 15), QPointF(20, 12))

    elif icon_name == "concatenate":
        # Stacked rectangles
        painter.drawRect(QRectF(5, 3, 14, 7))
        painter.drawRect(QRectF(5, 12, 14, 7))
        painter.setPen(QPen(accent, 1.5))
        painter.drawLine(QPointF(12, 10), QPointF(12, 12))
        painter.drawLine(QPointF(10, 11), QPointF(14, 11))

    elif icon_name == "aggregate":
        # Sigma symbol Σ
        painter.setPen(QPen(accent, 2.5))
        painter.drawLine(QPointF(6, 4), QPointF(18, 4))
        painter.drawLine(QPointF(6, 4), QPointF(12, 12))
        painter.drawLine(QPointF(12, 12), QPointF(6, 20))
        painter.drawLine(QPointF(6, 20), QPointF(18, 20))

    elif icon_name == "groupby":
        # Grouped dots
        painter.setBrush(accent)
        painter.drawEllipse(QRectF(3, 4, 5, 5))
        painter.drawEllipse(QRectF(3, 14, 5, 5))
        painter.setBrush(accent.lighter(140))
        painter.drawEllipse(QRectF(10, 4, 5, 5))
        painter.drawEllipse(QRectF(10, 14, 5, 5))
        painter.setBrush(accent.lighter(170))
        painter.drawEllipse(QRectF(17, 9, 5, 5))

    elif icon_name == "pivot":
        # Grid with rotation arrow
        painter.drawRect(QRectF(3, 3, 18, 18))
        painter.drawLine(QPointF(3, 10), QPointF(21, 10))
        painter.drawLine(QPointF(10, 3), QPointF(10, 21))
        painter.setPen(QPen(accent, 1.5))
        painter.drawArc(QRectF(13, 13, 6, 6), 0, 270 * 16)

    elif icon_name == "preprocess":
        # Gear / cog
        painter.drawEllipse(QRectF(7, 7, 10, 10))
        for angle in range(0, 360, 45):
            import math
            rad = math.radians(angle)
            cx, cy = 12, 12
            painter.drawLine(
                QPointF(cx + 5 * math.cos(rad), cy + 5 * math.sin(rad)),
                QPointF(cx + 8 * math.cos(rad), cy + 8 * math.sin(rad)),
            )

    elif icon_name == "impute":
        # Fill/patch: dotted line becoming solid
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(4, 12), QPointF(11, 12))
        painter.setPen(QPen(accent, 2.0, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(11, 12), QPointF(20, 12))
        painter.drawLine(QPointF(17, 9), QPointF(20, 12))
        painter.drawLine(QPointF(17, 15), QPointF(20, 12))

    elif icon_name == "continuize":
        # Step function → smooth curve
        painter.setPen(QPen(QColor("#999"), 1.5, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(4, 18), QPointF(8, 18))
        painter.drawLine(QPointF(8, 12), QPointF(12, 12))
        painter.drawLine(QPointF(12, 6), QPointF(16, 6))
        painter.setPen(QPen(accent, 2.0))
        path = QPainterPath()
        path.moveTo(4, 18)
        path.cubicTo(QPointF(8, 18), QPointF(8, 12), QPointF(10, 12))
        path.cubicTo(QPointF(12, 12), QPointF(12, 6), QPointF(16, 6))
        painter.drawPath(path)

    elif icon_name == "discretize":
        # Smooth curve → step function
        painter.setPen(QPen(QColor("#999"), 1.5, Qt.PenStyle.DashLine))
        path = QPainterPath()
        path.moveTo(4, 18)
        path.cubicTo(QPointF(10, 18), QPointF(14, 6), QPointF(20, 6))
        painter.drawPath(path)
        painter.setPen(QPen(accent, 2.0))
        painter.drawLine(QPointF(4, 18), QPointF(8, 18))
        painter.drawLine(QPointF(8, 18), QPointF(8, 12))
        painter.drawLine(QPointF(8, 12), QPointF(14, 12))
        painter.drawLine(QPointF(14, 12), QPointF(14, 6))
        painter.drawLine(QPointF(14, 6), QPointF(20, 6))

    elif icon_name == "melt":
        # Wide → tall (horizontal rect to vertical)
        painter.setPen(QPen(QColor("#999"), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(QRectF(3, 8, 14, 6))
        painter.setPen(QPen(accent, 2.0))
        painter.drawRect(QRectF(14, 3, 6, 18))

    elif icon_name == "createclass":
        # Tag / label
        painter.setBrush(QColor("#e07020"))
        path = QPainterPath()
        path.moveTo(4, 6)
        path.lineTo(14, 6)
        path.lineTo(20, 12)
        path.lineTo(14, 18)
        path.lineTo(4, 18)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(QColor("white"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(6, 10, 4, 4))

    elif icon_name == "createinstance":
        # Plus sign in a box
        painter.drawRect(QRectF(4, 4, 16, 16))
        painter.drawLine(QPointF(12, 7), QPointF(12, 17))
        painter.drawLine(QPointF(7, 12), QPointF(17, 12))

    elif icon_name == "formula":
        # f(x) text
        painter.setPen(QPen(accent, 2.0))
        font = QFont("Times New Roman", 11, QFont.Weight.Bold)
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(QRectF(2, 2, 20, 20), Qt.AlignmentFlag.AlignCenter, "f(x)")

    elif icon_name == "python":
        # Python-like interleaving snakes
        painter.setPen(QPen(QColor("#3776AB"), 2.0))
        path1 = QPainterPath()
        path1.moveTo(8, 4)
        path1.cubicTo(QPointF(4, 4), QPointF(4, 8), QPointF(8, 8))
        path1.lineTo(16, 8)
        path1.cubicTo(QPointF(20, 8), QPointF(20, 12), QPointF(16, 12))
        painter.drawPath(path1)
        painter.setPen(QPen(QColor("#FFD43B"), 2.0))
        path2 = QPainterPath()
        path2.moveTo(16, 20)
        path2.cubicTo(QPointF(20, 20), QPointF(20, 16), QPointF(16, 16))
        path2.lineTo(8, 16)
        path2.cubicTo(QPointF(4, 16), QPointF(4, 12), QPointF(8, 12))
        painter.drawPath(path2)

    # --- Data widgets ---
    elif icon_name == "file":
        # Document with folded corner
        painter.drawLine(QPointF(5, 3), QPointF(15, 3))
        painter.drawLine(QPointF(15, 3), QPointF(19, 7))
        painter.drawLine(QPointF(19, 7), QPointF(19, 21))
        painter.drawLine(QPointF(19, 21), QPointF(5, 21))
        painter.drawLine(QPointF(5, 21), QPointF(5, 3))
        painter.drawLine(QPointF(15, 3), QPointF(15, 7))
        painter.drawLine(QPointF(15, 7), QPointF(19, 7))
        painter.drawLine(QPointF(8, 11), QPointF(16, 11))
        painter.drawLine(QPointF(8, 14), QPointF(16, 14))
        painter.drawLine(QPointF(8, 17), QPointF(13, 17))

    elif icon_name == "csv":
        # CSV text badge
        painter.drawRect(QRectF(3, 5, 18, 14))
        painter.setPen(QPen(accent, 1.8))
        font = QFont("Consolas", 7, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(3, 5, 18, 14), Qt.AlignmentFlag.AlignCenter, "CSV")

    elif icon_name == "dataset":
        # Database cylinder
        painter.drawEllipse(QRectF(4, 3, 16, 6))
        painter.drawLine(QPointF(4, 6), QPointF(4, 18))
        painter.drawLine(QPointF(20, 6), QPointF(20, 18))
        painter.drawArc(QRectF(4, 15, 16, 6), 0, -180 * 16)
        painter.setPen(QPen(accent.lighter(140), 1.2))
        painter.drawArc(QRectF(4, 9, 16, 6), 0, -180 * 16)

    elif icon_name == "table":
        # Data grid / table
        painter.drawRect(QRectF(3, 3, 18, 18))
        painter.drawLine(QPointF(3, 8), QPointF(21, 8))
        painter.drawLine(QPointF(3, 13), QPointF(21, 13))
        painter.drawLine(QPointF(3, 18), QPointF(21, 18))
        painter.drawLine(QPointF(9, 3), QPointF(9, 21))
        painter.drawLine(QPointF(15, 3), QPointF(15, 21))
        # Header fill
        painter.setBrush(QColor(224, 112, 32, 60))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(3, 3, 18, 5))

    elif icon_name == "paint":
        # Paint brush
        painter.drawLine(QPointF(6, 18), QPointF(10, 10))
        painter.drawLine(QPointF(10, 10), QPointF(18, 4))
        painter.drawLine(QPointF(18, 4), QPointF(20, 6))
        painter.drawLine(QPointF(20, 6), QPointF(12, 12))
        painter.drawLine(QPointF(12, 12), QPointF(8, 20))
        painter.drawLine(QPointF(8, 20), QPointF(6, 18))
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(4, 17, 5, 5))

    elif icon_name == "info":
        # Info circle with "i"
        painter.drawEllipse(QRectF(3, 3, 18, 18))
        painter.setPen(QPen(accent, 2.0))
        font = QFont("Georgia", 11, QFont.Weight.Bold)
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(QRectF(3, 3, 18, 18), Qt.AlignmentFlag.AlignCenter, "i")

    elif icon_name == "rank":
        # Podium / bar chart ascending
        painter.setBrush(accent.lighter(160))
        painter.drawRect(QRectF(3, 14, 5, 7))
        painter.setBrush(accent)
        painter.drawRect(QRectF(10, 8, 5, 13))
        painter.setBrush(accent.lighter(130))
        painter.drawRect(QRectF(17, 3, 5, 18))

    elif icon_name == "edit":
        # Pencil
        painter.drawLine(QPointF(5, 19), QPointF(8, 16))
        painter.drawLine(QPointF(8, 16), QPointF(18, 6))
        painter.drawLine(QPointF(18, 6), QPointF(20, 8))
        painter.drawLine(QPointF(20, 8), QPointF(10, 18))
        painter.drawLine(QPointF(10, 18), QPointF(5, 19))
        painter.drawLine(QPointF(16, 8), QPointF(18, 10))

    elif icon_name == "color":
        # Color palette circles
        painter.setBrush(QColor("#e07020"))
        painter.drawEllipse(QRectF(3, 3, 8, 8))
        painter.setBrush(QColor("#3b82f6"))
        painter.drawEllipse(QRectF(13, 3, 8, 8))
        painter.setBrush(QColor("#22c55e"))
        painter.drawEllipse(QRectF(3, 13, 8, 8))
        painter.setBrush(QColor("#a855f7"))
        painter.drawEllipse(QRectF(13, 13, 8, 8))

    elif icon_name == "stats":
        # Bar chart with trend line
        painter.setBrush(accent.lighter(150))
        painter.drawRect(QRectF(3, 12, 4, 9))
        painter.drawRect(QRectF(8, 8, 4, 13))
        painter.drawRect(QRectF(13, 5, 4, 16))
        painter.drawRect(QRectF(18, 10, 4, 11))
        painter.setPen(QPen(QColor("#dc2626"), 1.8))
        painter.drawLine(QPointF(5, 11), QPointF(10, 7))
        painter.drawLine(QPointF(10, 7), QPointF(15, 4))
        painter.drawLine(QPointF(15, 4), QPointF(20, 9))

    elif icon_name == "save":
        # Floppy disk
        painter.drawRect(QRectF(3, 3, 18, 18))
        painter.drawRect(QRectF(7, 3, 8, 7))
        painter.drawRect(QRectF(6, 14, 12, 7))
        painter.drawLine(QPointF(9, 16), QPointF(15, 16))
        painter.drawLine(QPointF(9, 18), QPointF(15, 18))

    else:
        drawn = False

    painter.end()
    return QIcon(pixmap) if drawn else None


def get_widget_icon(icon_name: str):
    # Try custom drawn icon first
    custom = _make_custom_icon(icon_name)
    if custom is not None:
        return custom
    # Fall back to Qt standard icons
    app = QApplication.instance()
    style = app.style() if app is not None else None
    if style is None:
        return None
    return style.standardIcon(ICON_MAP.get(icon_name, QStyle.StandardPixmap.SP_FileIcon))


def get_toolbar_icon(icon_name: str) -> QIcon:
    pixmap = QPixmap(20, 20)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#5a4a36"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    if icon_name in {"zoom_in", "zoom_out"}:
        painter.drawEllipse(QRectF(3.5, 3.5, 9, 9))
        painter.drawLine(QPointF(11.2, 11.2), QPointF(16.5, 16.5))
        painter.drawLine(QPointF(6.4, 8.0), QPointF(9.6, 8.0))
        if icon_name == "zoom_in":
            painter.drawLine(QPointF(8.0, 6.4), QPointF(8.0, 9.6))
    elif icon_name == "pan":
        painter.drawLine(QPointF(10, 2.5), QPointF(10, 17.5))
        painter.drawLine(QPointF(2.5, 10), QPointF(17.5, 10))
        painter.drawLine(QPointF(10, 2.5), QPointF(7.5, 5.0))
        painter.drawLine(QPointF(10, 2.5), QPointF(12.5, 5.0))
        painter.drawLine(QPointF(10, 17.5), QPointF(7.5, 15.0))
        painter.drawLine(QPointF(10, 17.5), QPointF(12.5, 15.0))
        painter.drawLine(QPointF(2.5, 10), QPointF(5.0, 7.5))
        painter.drawLine(QPointF(2.5, 10), QPointF(5.0, 12.5))
        painter.drawLine(QPointF(17.5, 10), QPointF(15.0, 7.5))
        painter.drawLine(QPointF(17.5, 10), QPointF(15.0, 12.5))
    elif icon_name == "reset":
        painter.drawArc(QRectF(3.5, 3.5, 13, 13), 35 * 16, 288 * 16)
        painter.drawLine(QPointF(13.9, 3.7), QPointF(16.6, 4.2))
        painter.drawLine(QPointF(13.9, 3.7), QPointF(15.1, 6.1))
    elif icon_name == "text":
        painter.drawLine(QPointF(4.0, 4.5), QPointF(16.0, 4.5))
        painter.drawLine(QPointF(10.0, 4.5), QPointF(10.0, 16.0))
    elif icon_name == "arrow":
        painter.drawLine(QPointF(4.0, 15.5), QPointF(15.5, 4.0))
        painter.drawLine(QPointF(10.5, 4.0), QPointF(15.5, 4.0))
        painter.drawLine(QPointF(15.5, 4.0), QPointF(15.5, 9.0))
    elif icon_name == "expand":
        painter.drawLine(QPointF(6.0, 8.0), QPointF(10.0, 12.0))
        painter.drawLine(QPointF(10.0, 12.0), QPointF(14.0, 8.0))
    elif icon_name == "collapse":
        painter.drawLine(QPointF(6.0, 12.0), QPointF(10.0, 8.0))
        painter.drawLine(QPointF(10.0, 8.0), QPointF(14.0, 12.0))
    elif icon_name == "menu":
        painter.drawLine(QPointF(4.0, 6.0), QPointF(16.0, 6.0))
        painter.drawLine(QPointF(4.0, 10.0), QPointF(16.0, 10.0))
        painter.drawLine(QPointF(4.0, 14.0), QPointF(16.0, 14.0))
    elif icon_name == "help":
        painter.drawEllipse(QRectF(3.5, 3.5, 13.0, 13.0))
        painter.drawText(QRectF(3.5, 2.5, 13.0, 14.0), Qt.AlignmentFlag.AlignCenter, "?")
    elif icon_name == "report":
        painter.drawRect(QRectF(4.0, 3.5, 12.0, 14.0))
        painter.drawLine(QPointF(7.0, 7.0), QPointF(13.0, 7.0))
        painter.drawLine(QPointF(7.0, 10.0), QPointF(13.0, 10.0))
        painter.drawLine(QPointF(7.0, 13.0), QPointF(13.0, 13.0))
    elif icon_name == "data":
        painter.drawRect(QRectF(3.5, 4.0, 13.0, 12.0))
        painter.drawLine(QPointF(3.5, 8.0), QPointF(16.5, 8.0))
        painter.drawLine(QPointF(7.8, 4.0), QPointF(7.8, 16.0))
        painter.drawLine(QPointF(12.2, 4.0), QPointF(12.2, 16.0))
    elif icon_name == "preview":
        painter.drawRect(QRectF(3.5, 4.0, 13.0, 12.0))
        painter.drawLine(QPointF(3.5, 8.0), QPointF(16.5, 8.0))
        painter.drawLine(QPointF(9.8, 4.0), QPointF(9.8, 16.0))
    elif icon_name == "folder":
        painter.drawLine(QPointF(3.0, 7.0), QPointF(7.5, 7.0))
        painter.drawLine(QPointF(7.5, 7.0), QPointF(9.0, 5.0))
        painter.drawLine(QPointF(9.0, 5.0), QPointF(17.0, 5.0))
        painter.drawLine(QPointF(3.0, 7.0), QPointF(3.0, 15.0))
        painter.drawLine(QPointF(3.0, 15.0), QPointF(17.0, 15.0))
        painter.drawLine(QPointF(17.0, 15.0), QPointF(17.0, 5.0))
    elif icon_name == "reload":
        painter.drawArc(QRectF(4.0, 4.0, 12.0, 12.0), 40 * 16, 280 * 16)
        painter.drawLine(QPointF(13.7, 4.2), QPointF(16.0, 4.8))
        painter.drawLine(QPointF(13.7, 4.2), QPointF(14.9, 6.4))

    painter.end()
    return QIcon(pixmap)
