from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
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


def get_widget_icon(icon_name: str):
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
