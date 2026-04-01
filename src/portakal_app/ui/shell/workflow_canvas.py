from __future__ import annotations

import uuid
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPainterPathStroker, QPen, QTextCursor, QTransform
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import PortDefinition, WidgetDefinition, workflow_ports_are_compatible
from portakal_app.ui import i18n
from portakal_app.ui.icons import get_widget_icon


class ChannelSelectionDialog(QDialog):
    """Orange-style dialog for selecting which output channel to send through a cable."""

    def __init__(
        self,
        source_label: str,
        target_label: str,
        channels: tuple[str, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumWidth(340)
        self._selected_channel = channels[0]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(f"<b>{i18n.t('Edit Links')}</b>")
        layout.addWidget(title)

        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{source_label}</b>"))
        header.addStretch(1)
        header.addWidget(QLabel(f"<b>{target_label}</b>"))
        layout.addLayout(header)

        self._radios: list[QRadioButton] = []
        for i, ch in enumerate(channels):
            radio = QRadioButton(ch)
            if i == 0:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, c=ch: self._on_toggled(checked, c))
            self._radios.append(radio)
            layout.addWidget(radio)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet("""
            ChannelSelectionDialog {
                background: #faf6f0;
                border: 1px solid #c8bfb0;
                border-radius: 8px;
            }
        """)

    def _on_toggled(self, checked: bool, channel: str) -> None:
        if checked:
            self._selected_channel = channel

    def selected_channel(self) -> str:
        return self._selected_channel


@dataclass(frozen=True)
class WorkflowPortRef:
    node_id: str
    widget_id: str
    direction: str
    port_id: str


@dataclass
class WorkflowNodeRecord:
    node_id: str
    widget_id: str
    label: str


class WorkflowEdgeItem(QGraphicsPathItem):
    def __init__(
        self,
        source_item: "WorkflowNodeItem",
        source_port_id: str,
        target_item: "WorkflowNodeItem",
        target_port_id: str,
        channel: str = "",
        input_channel: str = "",
    ) -> None:
        super().__init__()
        self.source_item = source_item
        self.source_port_id = source_port_id
        self.target_item = target_item
        self.target_port_id = target_port_id
        self.channel = channel
        self.input_channel = input_channel
        self.setFlags(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(0)
        self._label_item = QGraphicsTextItem(self)
        self._label_item.setDefaultTextColor(QColor("#7e8ea0"))
        font = self._label_item.font()
        font.setPointSizeF(7.0)
        self._label_item.setFont(font)
        self.update_path()

    def update_path(self) -> None:
        source = self.source_item.port_scene_position("output", self.source_port_id)
        target = self.target_item.port_scene_position("input", self.target_port_id)
        delta = max(60.0, abs(target.x() - source.x()) / 2)
        path = QPainterPath(source)
        path.cubicTo(source.x() + delta, source.y(), target.x() - delta, target.y(), target.x(), target.y())
        self.setPath(path)
        self._sync_pen()
        self._update_label()

    def _update_label(self) -> None:
        parts: list[str] = []
        if self.channel:
            parts.append(self.channel)
        if self.input_channel:
            if parts:
                parts.append("→")
            parts.append(self.input_channel)
        label_text = " ".join(parts)
        self._label_item.setPlainText(label_text)
        if label_text:
            mid = self.path().pointAtPercent(0.5)
            br = self._label_item.boundingRect()
            self._label_item.setPos(mid.x() - br.width() / 2, mid.y() - br.height() - 2)
            self._label_item.show()
        else:
            self._label_item.hide()

    def _sync_pen(self) -> None:
        color = QColor("#7e8ea0") if self.isSelected() else QColor("#9aa8b5")
        self.setPen(QPen(color, 3))

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(self.path())

    def itemChange(self, change, value):
        if change == QGraphicsPathItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._sync_pen()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if not hasattr(scene, "views") or not scene.views(): # type: ignore
            super().mouseDoubleClickEvent(event)
            return

        parent_widget = scene.views()[0] # type: ignore
        changed = False

        channels = self.source_item.widget_definition.output_channels
        if channels and len(channels) > 1:
            dialog = ChannelSelectionDialog(
                self.source_item.display_label,
                self.target_item.display_label,
                channels,
                parent=parent_widget,
            )
            for radio in dialog._radios:
                if radio.text() == self.channel:
                    radio.setChecked(True)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.channel = dialog.selected_channel()
                changed = True

        in_channels = self.target_item.widget_definition.input_channels
        if in_channels and len(in_channels) > 1:
            multi = set(self.target_item.widget_definition.multi_input_channels)
            # Compute which channels are available for this edge:
            # current channel + channels not used by OTHER edges + multi channels
            used_by_others: set[str] = set()
            if isinstance(scene, WorkflowScene):
                for other_edge in scene._edges:
                    if other_edge is self:
                        continue
                    if (other_edge.target_item is self.target_item
                            and other_edge.target_port_id == self.target_port_id
                            and other_edge.input_channel):
                        used_by_others.add(other_edge.input_channel)
            available = tuple(
                ch for ch in in_channels
                if ch == self.input_channel   # always include current
                or ch not in used_by_others   # include if not used by others
                or ch in multi                # multi channels always available
            )
            if not available:
                available = in_channels  # fallback
            dialog = ChannelSelectionDialog(
                self.source_item.display_label,
                self.target_item.display_label,
                available,
                parent=parent_widget,
            )
            for radio in dialog._radios:
                if radio.text() == self.input_channel:
                    radio.setChecked(True)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.input_channel = dialog.selected_channel()
                changed = True

        if changed:
            if hasattr(scene, "_notify_workflow_changed"):
                scene._notify_workflow_changed() # type: ignore
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class WorkflowPreviewEdgeItem(QGraphicsPathItem):
    def __init__(self) -> None:
        super().__init__()
        pen = QPen(QColor("#bfc9d3"), 2, Qt.PenStyle.DashLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setZValue(0)
        self.hide()

    def update_path(self, source: QPointF, target: QPointF) -> None:
        delta = max(50.0, abs(target.x() - source.x()) / 2)
        path = QPainterPath(source)
        path.cubicTo(source.x() + delta, source.y(), target.x() - delta, target.y(), target.x(), target.y())
        self.setPath(path)
        self.show()


class AnnotationTextItem(QGraphicsTextItem):
    def __init__(self, annotation: "WorkflowTextAnnotationItem", text: str) -> None:
        super().__init__(text, annotation)
        self._annotation = annotation

    def focusInEvent(self, event) -> None:
        self._annotation._editing = True
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._annotation._editing = False
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:
        view = None
        scroll_values = None
        scene = self.scene()
        if scene is not None and scene.views():
            view = scene.views()[0]
            scroll_values = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
        super().keyPressEvent(event)
        self._annotation._fit_to_text()
        if view is not None and scroll_values is not None:
            view.horizontalScrollBar().setValue(scroll_values[0])
            view.verticalScrollBar().setValue(scroll_values[1])


class WorkflowTextAnnotationItem(QGraphicsObject):
    MIN_WIDTH = 96.0
    MIN_HEIGHT = 52.0
    PADDING = 12.0
    HANDLE = 10.0

    def __init__(self, text: str = "Text annotation", size: QSize | None = None) -> None:
        super().__init__()
        self._rect = QRectF(0, 0, float(size.width()) if size else 128.0, float(size.height()) if size else 58.0)
        self._resize_mode: str | None = None
        self._drag_origin = QPointF()
        self._start_rect = QRectF()
        self._start_pos = QPointF()
        self._editing = False
        self.setZValue(2)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self._text_item = AnnotationTextItem(self, text)
        self._text_item.setDefaultTextColor(QColor("#5f4d39"))
        self._text_item.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self._text_item.document().contentsChanged.connect(self._fit_to_text)
        self._sync_text_layout()
        self._fit_to_text()

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-2, -2, 2, 2)

    def toPlainText(self) -> str:
        return self._text_item.toPlainText()

    def begin_editing(self) -> None:
        self._editing = True
        self._text_item.setFocus(Qt.FocusReason.OtherFocusReason)
        cursor = self._text_item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text_item.setTextCursor(cursor)

    def annotation_size(self) -> QSize:
        return QSize(int(self._rect.width()), int(self._rect.height()))

    def is_editing(self) -> bool:
        return self._editing

    def _sync_text_layout(self) -> None:
        self._text_item.setPos(self.PADDING, self.PADDING - 2)
        self._text_item.setTextWidth(max(60.0, self._rect.width() - (self.PADDING * 2)))
        self.update()

    def _fit_to_text(self) -> None:
        document = self._text_item.document()
        view = None
        scroll_values = None
        scene = self.scene()
        if scene is not None and scene.views():
            view = scene.views()[0]
            scroll_values = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
        content_width = max(56.0, self._rect.width() - (self.PADDING * 2))
        document.setTextWidth(content_width)
        document.setPageSize(QSizeF(content_width, 100000.0))
        document.adjustSize()
        content_height = max(22.0, document.documentLayout().documentSize().height())
        new_width = self._rect.width()
        new_height = max(self.MIN_HEIGHT, content_height + (self.PADDING * 2) + 4.0)

        if abs(new_width - self._rect.width()) > 0.5 or abs(new_height - self._rect.height()) > 0.5:
            self.prepareGeometryChange()
            self._rect = QRectF(0, 0, new_width, new_height)
        self._sync_text_layout()
        if view is not None and scroll_values is not None:
            view.horizontalScrollBar().setValue(scroll_values[0])
            view.verticalScrollBar().setValue(scroll_values[1])

    def _resize_mode_at(self, pos: QPointF) -> str | None:
        if not self.isSelected():
            return None
        near_left = pos.x() <= self.HANDLE
        near_right = pos.x() >= self._rect.width() - self.HANDLE
        near_top = pos.y() <= self.HANDLE
        near_bottom = pos.y() >= self._rect.height() - self.HANDLE
        if near_left and near_top:
            return "tl"
        if near_right and near_top:
            return "tr"
        if near_left and near_bottom:
            return "bl"
        if near_right and near_bottom:
            return "br"
        if near_left:
            return "l"
        if near_right:
            return "r"
        if near_top:
            return "t"
        if near_bottom:
            return "b"
        return None

    def paint(self, painter: QPainter, _option, _widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = QColor("#fffdf9")
        border = QColor("#cfa56b") if self.isSelected() else QColor("#d9cfbe")
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.6))
        painter.drawRoundedRect(self._rect, 14, 14)
        if self.isSelected():
            painter.setBrush(QColor("#e2a952"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in (
                QPointF(self._rect.left(), self._rect.top()),
                QPointF(self._rect.right(), self._rect.top()),
                QPointF(self._rect.left(), self._rect.bottom()),
                QPointF(self._rect.right(), self._rect.bottom()),
            ):
                painter.drawEllipse(point, 3.4, 3.4)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            mode = self._resize_mode_at(event.pos())
            if mode is not None:
                self._resize_mode = mode
                self._drag_origin = event.scenePos()
                self._start_rect = QRectF(self._rect)
                self._start_pos = QPointF(self.pos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resize_mode is None:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._drag_origin
        width = self._start_rect.width()
        height = self._start_rect.height()
        new_pos = QPointF(self._start_pos)
        if "l" in self._resize_mode:
            proposed = self._start_rect.width() - delta.x()
            width = max(self.MIN_WIDTH, proposed)
            new_pos.setX(self._start_pos.x() + (self._start_rect.width() - width))
        if "r" in self._resize_mode:
            width = max(self.MIN_WIDTH, self._start_rect.width() + delta.x())
        if "t" in self._resize_mode:
            proposed = self._start_rect.height() - delta.y()
            height = max(self.MIN_HEIGHT, proposed)
            new_pos.setY(self._start_pos.y() + (self._start_rect.height() - height))
        if "b" in self._resize_mode:
            height = max(self.MIN_HEIGHT, self._start_rect.height() + delta.y())
        self.prepareGeometryChange()
        self._rect = QRectF(0, 0, width, height)
        self.setPos(new_pos)
        self._sync_text_layout()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._resize_mode = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.begin_editing()
        event.accept()


class WorkflowArrowAnnotationItem(QGraphicsObject):
    HANDLE_RADIUS = 6.0

    def __init__(self, delta: QPointF | None = None) -> None:
        super().__init__()
        self._delta = delta or QPointF(140, 0)
        self._adjusting = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)

    @property
    def delta(self) -> QPointF:
        return QPointF(self._delta)

    def boundingRect(self) -> QRectF:
        left = min(0.0, self._delta.x())
        top = min(0.0, self._delta.y())
        right = max(0.0, self._delta.x())
        bottom = max(0.0, self._delta.y())
        return QRectF(left, top, right - left, bottom - top).adjusted(-14, -14, 14, 14)

    def _arrow_path(self) -> QPainterPath:
        line_end = QPointF(self._delta.x(), self._delta.y())
        path = QPainterPath(QPointF(0, 0))
        path.lineTo(line_end)
        arrow_size = 14.0
        length = max((line_end.x() ** 2 + line_end.y() ** 2) ** 0.5, 1.0)
        unit_x = line_end.x() / length
        unit_y = line_end.y() / length
        left = QPointF(
            line_end.x() - unit_x * arrow_size - unit_y * arrow_size * 0.5,
            line_end.y() - unit_y * arrow_size + unit_x * arrow_size * 0.5,
        )
        right = QPointF(
            line_end.x() - unit_x * arrow_size + unit_y * arrow_size * 0.5,
            line_end.y() - unit_y * arrow_size - unit_x * arrow_size * 0.5,
        )
        path.moveTo(line_end)
        path.lineTo(left)
        path.moveTo(line_end)
        path.lineTo(right)
        return path

    def paint(self, painter: QPainter, _option, _widget: QWidget | None = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#d57b34") if self.isSelected() else QColor("#c7a16f")
        painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(self._arrow_path())
        if self.isSelected():
            painter.setBrush(QColor("#e2a952"))
            painter.setPen(QPen(QColor("#ffffff"), 1.4))
            painter.drawEllipse(QPointF(0, 0), self.HANDLE_RADIUS - 1, self.HANDLE_RADIUS - 1)
            painter.drawEllipse(self._delta, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

    def _end_handle_hit(self, pos: QPointF) -> bool:
        dx = pos.x() - self._delta.x()
        dy = pos.y() - self._delta.y()
        return (dx * dx) + (dy * dy) <= (self.HANDLE_RADIUS + 4) ** 2

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._end_handle_hit(event.pos()):
            self._adjusting = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._adjusting:
            end = event.pos()
            if abs(end.x()) < 8 and abs(end.y()) < 8:
                end = QPointF(80, 0)
            self.prepareGeometryChange()
            self._delta = QPointF(end)
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._adjusting = False
        super().mouseReleaseEvent(event)


class WorkflowNodeItem(QGraphicsObject):
    activated = Signal(str)
    portPressed = Signal(object)
    moved = Signal()

    WIDTH = 108.0
    HEIGHT = 58.0
    PORT_RADIUS = 5.0
    PORT_HIT_RADIUS = 12.0
    PORT_EVENT_MARGIN = 16.0
    PORT_EDGE_OFFSET = 1.0
    DELETE_BUTTON_SIZE = 18.0

    def __init__(self, widget_definition: WidgetDefinition, node_id: str | None = None, label: str | None = None) -> None:
        super().__init__()
        self.node_id = node_id or str(uuid.uuid4())
        self.widget_definition = widget_definition
        self.display_label = label or widget_definition.label
        self._icon: QIcon | None = get_widget_icon(widget_definition.icon_name)
        self._edges: list[WorkflowEdgeItem] = []
        self._active_port_drag: WorkflowPortRef | None = None
        self._delete_hovered = False
        self.setFlags(
            QGraphicsObject.GraphicsItemFlag.ItemIsMovable
            | QGraphicsObject.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        return self._body_rect().adjusted(-self.PORT_EVENT_MARGIN, 0.0, self.PORT_EVENT_MARGIN, 0.0)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(self._body_rect(), 28, 28)
        for direction, ports in (("input", self.widget_definition.input_ports), ("output", self.widget_definition.output_ports)):
            for index, _port in enumerate(ports):
                path.addEllipse(self._port_center(direction, index, len(ports)), self.PORT_HIT_RADIUS, self.PORT_HIT_RADIUS)
        if self.isSelected():
            path.addEllipse(self._delete_button_rect())
        return path

    def add_edge(self, edge: WorkflowEdgeItem) -> None:
        self._edges.append(edge)

    def remove_edge(self, edge: WorkflowEdgeItem) -> None:
        if edge in self._edges:
            self._edges.remove(edge)

    def set_display_label(self, label: str) -> None:
        self.display_label = label or self.widget_definition.label
        self.update()

    def update_widget_definition(self, widget_definition: WidgetDefinition) -> None:
        previous_default_label = self.widget_definition.label
        self.widget_definition = widget_definition
        self._icon = get_widget_icon(widget_definition.icon_name)
        if self.display_label == previous_default_label:
            self.display_label = widget_definition.label
        self.update()

    def center_anchor(self) -> QPointF:
        return self.mapToScene(self._body_rect().center())

    def _body_rect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, self.HEIGHT)

    def _ports(self, direction: str) -> tuple[PortDefinition, ...]:
        return self.widget_definition.input_ports if direction == "input" else self.widget_definition.output_ports

    def port_scene_position(self, direction: str, port_id: str) -> QPointF:
        return self.mapToScene(self._port_local_center(direction, port_id))

    def _port_local_center(self, direction: str, port_id: str) -> QPointF:
        ports = self._ports(direction)
        for index, port in enumerate(ports):
            if port.id == port_id:
                return self._port_center(direction, index, len(ports))
        raise KeyError(f"Unknown {direction} port {port_id} for {self.widget_definition.id}")

    def _port_center(self, direction: str, index: int, count: int) -> QPointF:
        spacing = self.HEIGHT / (count + 1)
        y = spacing * (index + 1)
        x = -self.PORT_EDGE_OFFSET if direction == "input" else self.WIDTH + self.PORT_EDGE_OFFSET
        return QPointF(x, y)

    def _delete_button_rect(self) -> QRectF:
        size = self.DELETE_BUTTON_SIZE
        return QRectF(self.WIDTH - size - 6.0, 6.0, size, size)

    def _hit_test_port(self, position: QPointF) -> WorkflowPortRef | None:
        for direction, ports in (("input", self.widget_definition.input_ports), ("output", self.widget_definition.output_ports)):
            for index, port in enumerate(ports):
                center = self._port_center(direction, index, len(ports))
                dx = position.x() - center.x()
                dy = position.y() - center.y()
                if (dx * dx) + (dy * dy) <= self.PORT_HIT_RADIUS ** 2:
                    return WorkflowPortRef(self.node_id, self.widget_definition.id, direction, port.id)
        return None

    def paint(self, painter: QPainter, _option, _widget: QWidget | None = None) -> None:
        rect = self._body_rect()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        border_color = QColor("#d0d8df") if not self.isSelected() else QColor("#e3a849")
        painter.setBrush(QBrush(QColor("#fffefb")))
        painter.setPen(QPen(border_color, 1.6))
        painter.drawRoundedRect(rect, 28, 28)

        metrics = painter.fontMetrics()
        text_width = min(metrics.horizontalAdvance(self.display_label), int(rect.width() - 44))
        gap = 8.0
        group_width = 24.0 + gap + text_width
        group_left = max(10.0, (rect.width() - group_width) / 2)

        icon_circle = QRectF(group_left, 17, 24, 24)
        painter.setBrush(QColor("#ffd9ab"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(icon_circle)

        if self._icon is not None:
            self._icon.paint(painter, icon_circle.toRect().adjusted(4, 4, -4, -4))

        painter.setPen(QPen(QColor("#24313b")))
        text_rect = QRectF(icon_circle.right() + gap, 18, text_width + 4, 22)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.display_label)

        for direction, ports, color in (
            ("input", self.widget_definition.input_ports, QColor("#4d95da")),
            ("output", self.widget_definition.output_ports, QColor("#e3a849")),
        ):
            for index, _port in enumerate(ports):
                center = self._port_center(direction, index, len(ports))
                painter.setBrush(color)
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawEllipse(center, self.PORT_RADIUS, self.PORT_RADIUS)

        if self.isSelected():
            delete_rect = self._delete_button_rect()
            painter.setBrush(QColor("#f0b2a7") if self._delete_hovered else QColor("#f6d7d1"))
            painter.setPen(QPen(QColor("#c15b4d"), 1.2))
            painter.drawEllipse(delete_rect)
            painter.setPen(QPen(QColor("#7a2418"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(delete_rect.left() + 5.0, delete_rect.top() + 5.0, delete_rect.right() - 5.0, delete_rect.bottom() - 5.0)
            painter.drawLine(delete_rect.right() - 5.0, delete_rect.top() + 5.0, delete_rect.left() + 5.0, delete_rect.bottom() - 5.0)

    def mousePressEvent(self, event) -> None:
        if self.isSelected() and self._delete_button_rect().contains(event.pos()):
            scene = self.scene()
            if isinstance(scene, WorkflowScene):
                scene.delete_node(self)
            event.accept()
            return
        port_ref = self._hit_test_port(event.pos())
        if port_ref is not None:
            self._active_port_drag = port_ref if port_ref.direction == "output" else None
            self.portPressed.emit(port_ref)
            event.accept()
            return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        hovered = self.isSelected() and self._delete_button_rect().contains(event.pos())
        if hovered != self._delete_hovered:
            self._delete_hovered = hovered
            self.update(self._delete_button_rect())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._delete_hovered:
            self._delete_hovered = False
            self.update(self._delete_button_rect())
        super().hoverLeaveEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._active_port_drag is not None and self._active_port_drag.direction == "output":
            scene = self.scene()
            if isinstance(scene, WorkflowScene):
                scene.drag_connection_to(self.mapToScene(event.pos()))
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.activated.emit(self.node_id)
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._active_port_drag is not None and self._active_port_drag.direction == "output":
            scene = self.scene()
            if isinstance(scene, WorkflowScene):
                scene.finish_connection_drag_at(self.mapToScene(event.pos()))
            self._active_port_drag = None
            event.accept()
            return
        self._active_port_drag = None
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsObject.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self._edges:
                edge.update_path()
            self.moved.emit()
        return super().itemChange(change, value)


class WorkflowScene(QGraphicsScene):
    nodeActivated = Signal(str)
    statusMessage = Signal(str)
    workflowChanged = Signal()
    SCENE_WIDTH = 7000
    SCENE_HEIGHT = 4500
    PORT_SCENE_HIT_RADIUS = 42.0

    def __init__(self, widget_index: dict[str, WidgetDefinition], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._widget_index = widget_index
        self._nodes: dict[str, WorkflowNodeItem] = {}
        self._edges: list[WorkflowEdgeItem] = []
        self._annotations: list[QGraphicsObject] = []
        self._edge_keys: set[tuple[str, str, str, str]] = set()
        self._occupied_inputs: set[tuple[str, str]] = set()
        self._pending_output: WorkflowPortRef | None = None
        self._preview_edge = WorkflowPreviewEdgeItem()
        self._batch_depth = 0
        self.addItem(self._preview_edge)
        self.setSceneRect(0, 0, self.SCENE_WIDTH, self.SCENE_HEIGHT)

    def add_node(
        self,
        widget_id: str,
        position: QPointF,
        node_id: str | None = None,
        label: str | None = None,
        emit_status: bool = True,
    ) -> WorkflowNodeRecord:
        widget_definition = self._widget_index[widget_id]
        node = WorkflowNodeItem(widget_definition, node_id=node_id, label=label)
        node.setPos(position)
        node.activated.connect(self.nodeActivated.emit)
        node.portPressed.connect(self._handle_port_pressed)
        self.addItem(node)
        self._nodes[node.node_id] = node
        self.clearSelection()
        node.setSelected(True)
        if emit_status:
            self.statusMessage.emit(f"{widget_definition.label} added to the workflow.")
        self._notify_workflow_changed()
        return WorkflowNodeRecord(node.node_id, widget_id, node.display_label)

    def update_widget_index(self, widget_index: dict[str, WidgetDefinition]) -> None:
        self._widget_index = widget_index
        for node in self._nodes.values():
            widget_definition = widget_index.get(node.widget_definition.id)
            if widget_definition is None:
                continue
            node.update_widget_definition(widget_definition)
        self.update()

    def add_text_annotation(
        self,
        position: QPointF | None = None,
        text: str = "Text annotation",
        size: QSize | None = None,
    ) -> WorkflowTextAnnotationItem:
        annotation = WorkflowTextAnnotationItem(text)
        if size is not None:
            annotation = WorkflowTextAnnotationItem(text, size=size)
        annotation.setPos(position or QPointF(120, 120))
        self.addItem(annotation)
        self._annotations.append(annotation)
        self.clearSelection()
        annotation.setSelected(True)
        self.statusMessage.emit("Text annotation added.")
        self._notify_workflow_changed()
        annotation.begin_editing()
        return annotation

    def add_arrow_annotation(self, position: QPointF | None = None, delta: QPointF | None = None) -> WorkflowArrowAnnotationItem:
        annotation = WorkflowArrowAnnotationItem(delta)
        annotation.setPos(position or QPointF(120, 180))
        self.addItem(annotation)
        self._annotations.append(annotation)
        self.clearSelection()
        annotation.setSelected(True)
        self.statusMessage.emit("Arrow annotation added.")
        self._notify_workflow_changed()
        return annotation

    def create_connection(
        self,
        source_node_id: str,
        target_node_id: str,
        source_port_id: str | None = None,
        target_port_id: str | None = None,
        channel: str = "",
        input_channel: str = "",
        emit_status: bool = True,
    ) -> bool:
        source_node = self._nodes[source_node_id]
        target_node = self._nodes[target_node_id]
        if source_port_id is None:
            if not source_node.widget_definition.output_ports:
                if emit_status:
                    self.statusMessage.emit(f"{source_node.widget_definition.label} has no output ports.")
                return False
            source_port_id = source_node.widget_definition.output_ports[0].id
        if target_port_id is None:
            if not target_node.widget_definition.input_ports:
                if emit_status:
                    self.statusMessage.emit(f"{target_node.widget_definition.label} has no input ports.")
                return False
            target_port_id = target_node.widget_definition.input_ports[0].id

        source_ref = WorkflowPortRef(source_node_id, source_node.widget_definition.id, "output", source_port_id)
        target_ref = WorkflowPortRef(target_node_id, target_node.widget_definition.id, "input", target_port_id)
        success, message = self._create_connection_from_refs(
            source_ref, target_ref, channel=channel, input_channel=input_channel,
        )
        if emit_status:
            self.statusMessage.emit(message)
        return success

    def begin_connection_drag(self, source_node_id: str, source_port_id: str | None = None) -> bool:
        source_node = self._nodes[source_node_id]
        if source_port_id is None:
            if not source_node.widget_definition.output_ports:
                self.statusMessage.emit(f"{source_node.widget_definition.label} has no output ports.")
                return False
            source_port_id = source_node.widget_definition.output_ports[0].id
        source_ref = WorkflowPortRef(source_node_id, source_node.widget_definition.id, "output", source_port_id)
        self._handle_port_pressed(source_ref)
        return True

    def drag_connection_to(self, scene_pos: QPointF) -> None:
        if self._pending_output is None:
            return
        source_node = self._nodes[self._pending_output.node_id]
        start = source_node.port_scene_position("output", self._pending_output.port_id)
        self._preview_edge.update_path(start, scene_pos)

    def finish_connection_drag_at(self, scene_pos: QPointF) -> bool:
        if self._pending_output is None:
            return False
        target_port = self.port_ref_at(scene_pos, direction="input")
        if target_port is None:
            self.statusMessage.emit("Connection cancelled.")
            self._clear_pending_connection()
            return False

        source_node = self._nodes[self._pending_output.node_id]
        target_node = self._nodes[target_port.node_id]
        parent_widget = self.views()[0] if self.views() else None
        channel = ""
        input_channel = ""

        channels = source_node.widget_definition.output_channels
        if channels and len(channels) > 1:
            dialog = ChannelSelectionDialog(
                source_node.display_label,
                target_node.display_label,
                channels,
                parent=parent_widget,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.statusMessage.emit("Connection cancelled.")
                self._clear_pending_connection()
                return False
            channel = dialog.selected_channel()

        in_channels = target_node.widget_definition.input_channels
        if in_channels and len(in_channels) > 1:
            used = self._used_input_channels(target_port.node_id, target_port.port_id)
            multi = set(target_node.widget_definition.multi_input_channels)
            # Multi channels stay available even if already used.
            available = tuple(
                ch for ch in in_channels
                if ch not in used or ch in multi
            )
            if not available:
                self.statusMessage.emit("All input channels are already connected.")
                self._clear_pending_connection()
                return False
            if len(available) == 1:
                input_channel = available[0]
            else:
                dialog = ChannelSelectionDialog(
                    source_node.display_label,
                    target_node.display_label,
                    available,
                    parent=parent_widget,
                )
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    self.statusMessage.emit("Connection cancelled.")
                    self._clear_pending_connection()
                    return False
                input_channel = dialog.selected_channel()

        success, message = self._create_connection_from_refs(
            self._pending_output, target_port, channel=channel, input_channel=input_channel,
        )
        self.statusMessage.emit(message)
        self._clear_pending_connection()
        return success

    def clear_workflow(self, emit_status: bool = True) -> None:
        self._begin_batch()
        try:
            for edge in list(self._edges):
                self._remove_edge(edge)
            for node in list(self._nodes.values()):
                self._remove_node(node)
            for annotation in list(self._annotations):
                self._remove_annotation(annotation)
            self._clear_pending_connection()
        finally:
            self._end_batch()
        if emit_status:
            self.statusMessage.emit("Workflow cleared.")
        self._notify_workflow_changed()

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "widget_id": node.widget_definition.id,
                    "label": node.display_label,
                    "x": float(node.pos().x()),
                    "y": float(node.pos().y()),
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source_node_id": edge.source_item.node_id,
                    "source_port_id": edge.source_port_id,
                    "target_node_id": edge.target_item.node_id,
                    "target_port_id": edge.target_port_id,
                    "channel": edge.channel,
                    "input_channel": edge.input_channel,
                }
                for edge in self._edges
            ],
            "annotations": [self._annotation_snapshot(annotation) for annotation in self._annotations],
        }

    def restore_snapshot(self, snapshot: dict[str, list[dict[str, object]]], emit_status: bool = True) -> None:
        self._begin_batch()
        try:
            self.clear_workflow(emit_status=False)
            for node_data in snapshot.get("nodes", []):
                self.add_node(
                    str(node_data["widget_id"]),
                    QPointF(float(node_data["x"]), float(node_data["y"])),
                    node_id=str(node_data["id"]),
                    label=str(node_data.get("label") or ""),
                    emit_status=False,
                )
            for edge_data in snapshot.get("edges", []):
                self.create_connection(
                    str(edge_data["source_node_id"]),
                    str(edge_data["target_node_id"]),
                    source_port_id=str(edge_data["source_port_id"]),
                    target_port_id=str(edge_data["target_port_id"]),
                    channel=str(edge_data.get("channel") or ""),
                    input_channel=str(edge_data.get("input_channel") or ""),
                    emit_status=False,
                )
            for annotation_data in snapshot.get("annotations", []):
                self._restore_annotation(annotation_data)
        finally:
            self._end_batch()
        if emit_status:
            self.statusMessage.emit("Workflow loaded.")
        self._notify_workflow_changed()

    def select_all_items(self) -> bool:
        selectable_items = [item for item in self.items() if item is not self._preview_edge]
        if not selectable_items:
            self.statusMessage.emit("Nothing to select.")
            return False
        for item in selectable_items:
            if item.flags() & item.GraphicsItemFlag.ItemIsSelectable:
                item.setSelected(True)
        self.statusMessage.emit("Selected all workflow items.")
        return True

    def selected_node_items(self) -> list[WorkflowNodeItem]:
        return [item for item in self.selectedItems() if isinstance(item, WorkflowNodeItem)]

    def primary_selected_node(self) -> WorkflowNodeItem | None:
        nodes = self.selected_node_items()
        return nodes[0] if nodes else None

    def rename_selected_node(self, label: str) -> bool:
        node = self.primary_selected_node()
        if node is None:
            self.statusMessage.emit("Select a widget to rename it.")
            return False
        node.set_display_label(label)
        self.statusMessage.emit(f"Renamed widget to {node.display_label}.")
        self._notify_workflow_changed()
        return True

    def copy_selection(self) -> dict[str, list[dict[str, object]]]:
        selected_nodes = {node.node_id: node for node in self.selected_node_items()}
        selected_annotations = [item for item in self.selectedItems() if item in self._annotations]
        selected_edges = [
            edge
            for edge in self.selectedItems()
            if isinstance(edge, WorkflowEdgeItem)
            and edge.source_item.node_id in selected_nodes
            and edge.target_item.node_id in selected_nodes
        ]
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "widget_id": node.widget_definition.id,
                    "label": node.display_label,
                    "x": float(node.pos().x()),
                    "y": float(node.pos().y()),
                }
                for node in selected_nodes.values()
            ],
            "edges": [
                {
                    "source_node_id": edge.source_item.node_id,
                    "source_port_id": edge.source_port_id,
                    "target_node_id": edge.target_item.node_id,
                    "target_port_id": edge.target_port_id,
                    "channel": edge.channel,
                    "input_channel": edge.input_channel,
                }
                for edge in selected_edges
            ],
            "annotations": [self._annotation_snapshot(annotation) for annotation in selected_annotations],
        }

    def paste_snapshot(self, fragment: dict[str, list[dict[str, object]]], offset: QPointF | None = None) -> bool:
        nodes = fragment.get("nodes", [])
        annotations = fragment.get("annotations", [])
        if not nodes and not annotations:
            self.statusMessage.emit("Nothing to paste.")
            return False

        offset = offset or QPointF(28, 28)
        node_id_map: dict[str, str] = {}
        self._begin_batch()
        try:
            self.clearSelection()
            for node_data in nodes:
                record = self.add_node(
                    str(node_data["widget_id"]),
                    QPointF(float(node_data["x"]) + offset.x(), float(node_data["y"]) + offset.y()),
                    label=str(node_data.get("label") or ""),
                    emit_status=False,
                )
                node_id_map[str(node_data["id"])] = record.node_id
            for edge_data in fragment.get("edges", []):
                source_id = node_id_map.get(str(edge_data["source_node_id"]))
                target_id = node_id_map.get(str(edge_data["target_node_id"]))
                if source_id and target_id:
                    self.create_connection(
                        source_id,
                        target_id,
                        source_port_id=str(edge_data["source_port_id"]),
                        target_port_id=str(edge_data["target_port_id"]),
                        channel=str(edge_data.get("channel") or ""),
                        input_channel=str(edge_data.get("input_channel") or ""),
                        emit_status=False,
                    )
            for annotation_data in annotations:
                self._restore_annotation(annotation_data, offset=offset)
        finally:
            self._end_batch()
        item_count = len(nodes) + len(annotations)
        self.statusMessage.emit(f"Pasted {item_count} item(s).")
        self._notify_workflow_changed()
        return True

    def duplicate_selected_items(self, offset: QPointF | None = None) -> bool:
        return self.paste_snapshot(self.copy_selection(), offset=offset or QPointF(42, 42))

    def _handle_port_pressed(self, port_ref: WorkflowPortRef) -> None:
        widget_definition = self._widget_index[port_ref.widget_id]
        port_label = self._port_label(port_ref)
        if port_ref.direction == "output":
            self._pending_output = port_ref
            start = self._nodes[port_ref.node_id].port_scene_position("output", port_ref.port_id)
            self._preview_edge.update_path(start, start)
            self.statusMessage.emit(f"{widget_definition.label}:{port_label} selected. Drag to a compatible input port.")
            return

        if self._pending_output is None:
            self.statusMessage.emit("Start a connection by dragging from an output port.")
            return

        success, message = self._create_connection_from_refs(self._pending_output, port_ref)
        self._clear_pending_connection()
        self.statusMessage.emit(message)

    def _create_connection_from_refs(
        self, source_ref: WorkflowPortRef, target_ref: WorkflowPortRef,
        channel: str = "", input_channel: str = "",
    ) -> tuple[bool, str]:
        valid, message = self._validate_connection(source_ref, target_ref)
        if not valid:
            return False, message

        source_node = self._nodes[source_ref.node_id]
        target_node = self._nodes[target_ref.node_id]

        if not channel:
            channels = source_node.widget_definition.output_channels
            if channels:
                channel = channels[0]

        if not input_channel:
            in_channels = target_node.widget_definition.input_channels
            if in_channels:
                input_channel = in_channels[0]

        edge = WorkflowEdgeItem(
            source_node, source_ref.port_id, target_node, target_ref.port_id,
            channel=channel, input_channel=input_channel,
        )
        source_node.add_edge(edge)
        target_node.add_edge(edge)
        self.addItem(edge)
        edge.update_path()
        self._edges.append(edge)
        self._edge_keys.add((source_ref.node_id, source_ref.port_id, target_ref.node_id, target_ref.port_id))
        self._occupied_inputs.add((target_ref.node_id, target_ref.port_id))
        self._notify_workflow_changed()
        return True, f"Connected {source_node.display_label} to {target_node.display_label}."

    def _clear_pending_connection(self) -> None:
        self._pending_output = None
        self._preview_edge.hide()
        self._preview_edge.setPath(QPainterPath())

    def _validate_connection(self, source_ref: WorkflowPortRef, target_ref: WorkflowPortRef) -> tuple[bool, str]:
        if source_ref.direction != "output":
            return False, "Connections must start from an output port."
        if target_ref.direction != "input":
            return False, "Connections must end at an input port."
        if source_ref.node_id == target_ref.node_id:
            return False, "A node cannot connect to itself."
        if not self._port_exists(source_ref) or not self._port_exists(target_ref):
            return False, "Unknown workflow port."
        edge_key = (source_ref.node_id, source_ref.port_id, target_ref.node_id, target_ref.port_id)
        if edge_key in self._edge_keys:
            return False, "These ports are already connected."
        target_node = self._nodes.get(target_ref.node_id)
        has_input_channels = target_node is not None and len(target_node.widget_definition.input_channels) > 1
        if (target_ref.node_id, target_ref.port_id) in self._occupied_inputs and not has_input_channels:
            return False, "That input port is already connected."
        if has_input_channels:
            used_channels = self._used_input_channels(target_ref.node_id, target_ref.port_id)
            multi = set(target_node.widget_definition.multi_input_channels)
            # Multi channels can accept unlimited connections; only exclusive
            # channels count as "used" for availability.
            available = set(target_node.widget_definition.input_channels) - (used_channels - multi)
            if not available:
                return False, "All input channels are already connected."
        source_label = self._port_label(source_ref)
        target_label = self._port_label(target_ref)
        if not workflow_ports_are_compatible(source_ref.widget_id, source_label, target_ref.widget_id, target_label):
            return False, f"Port types do not match: {source_label} cannot connect to {target_label}."
        return True, "Connection created."

    def _used_input_channels(self, node_id: str, port_id: str) -> set[str]:
        used: set[str] = set()
        for edge in self._edges:
            if edge.target_item.node_id == node_id and edge.target_port_id == port_id and edge.input_channel:
                used.add(edge.input_channel)
        return used

    def _port_exists(self, port_ref: WorkflowPortRef) -> bool:
        node = self._nodes.get(port_ref.node_id)
        if node is None:
            return False
        ports = node.widget_definition.input_ports if port_ref.direction == "input" else node.widget_definition.output_ports
        return any(port.id == port_ref.port_id for port in ports)

    def _port_label(self, port_ref: WorkflowPortRef) -> str:
        node = self._nodes[port_ref.node_id]
        ports = node.widget_definition.input_ports if port_ref.direction == "input" else node.widget_definition.output_ports
        return next(port.label for port in ports if port.id == port_ref.port_id)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def node_records(self) -> list[WorkflowNodeRecord]:
        return [WorkflowNodeRecord(node.node_id, node.widget_definition.id, node.display_label) for node in self._nodes.values()]

    def node_record(self, node_id: str) -> WorkflowNodeRecord | None:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        return WorkflowNodeRecord(node.node_id, node.widget_definition.id, node.display_label)

    def widget_has_data_path(self, widget_id: str) -> bool:
        target_node_ids = {node_id for node_id, node in self._nodes.items() if node.widget_definition.id == widget_id}
        if not target_node_ids:
            return False

        source_node_ids = {
            node_id
            for node_id, node in self._nodes.items()
            if not node.widget_definition.input_ports
            and any(port.label == "Data" for port in node.widget_definition.output_ports)
        }
        if not source_node_ids:
            return False
        if target_node_ids & source_node_ids:
            return True

        adjacency: dict[str, set[str]] = {}
        for edge in self._edges:
            adjacency.setdefault(edge.source_item.node_id, set()).add(edge.target_item.node_id)

        pending = list(source_node_ids)
        visited = set(source_node_ids)
        while pending:
            current = pending.pop()
            for neighbour in adjacency.get(current, set()):
                if neighbour in visited:
                    continue
                if neighbour in target_node_ids:
                    return True
                visited.add(neighbour)
                pending.append(neighbour)
        return False

    def delete_selected_items(self) -> bool:
        selected = self.selectedItems()
        if not selected:
            self.statusMessage.emit("Nothing selected to delete.")
            return False

        removed_nodes = 0
        removed_edges = 0
        removed_annotations = 0

        for item in list(selected):
            if isinstance(item, WorkflowNodeItem):
                removed_edges += self._remove_node(item)
                removed_nodes += 1
            elif isinstance(item, WorkflowEdgeItem):
                if self._remove_edge(item):
                    removed_edges += 1
            elif item in self._annotations:
                if self._remove_annotation(item):
                    removed_annotations += 1

        if removed_nodes or removed_edges or removed_annotations:
            parts: list[str] = []
            if removed_nodes:
                parts.append(f"{removed_nodes} node")
            if removed_edges:
                parts.append(f"{removed_edges} connection")
            if removed_annotations:
                parts.append(f"{removed_annotations} annotation")
            self.statusMessage.emit(f"Deleted {' and '.join(parts)}.")
            self._notify_workflow_changed()
            return True

        self.statusMessage.emit("Nothing selected to delete.")
        return False

    def delete_node(self, node: WorkflowNodeItem) -> bool:
        if node.node_id not in self._nodes:
            return False
        removed_edges = self._remove_node(node)
        parts = ["1 node"]
        if removed_edges:
            parts.append(f"{removed_edges} connection")
            if removed_edges != 1:
                parts[-1] += "s"
        self.statusMessage.emit(f"Deleted {' and '.join(parts)}.")
        self._notify_workflow_changed()
        return True

    def _remove_node(self, node: WorkflowNodeItem) -> int:
        removed_edges = 0
        for edge in list(node._edges):
            if self._remove_edge(edge):
                removed_edges += 1
        self._nodes.pop(node.node_id, None)
        if self._pending_output is not None and self._pending_output.node_id == node.node_id:
            self._clear_pending_connection()
        self.removeItem(node)
        return removed_edges

    def _remove_edge(self, edge: WorkflowEdgeItem) -> bool:
        if edge not in self._edges:
            return False
        self._edges.remove(edge)
        edge_key = (edge.source_item.node_id, edge.source_port_id, edge.target_item.node_id, edge.target_port_id)
        self._edge_keys.discard(edge_key)
        self._occupied_inputs.discard((edge.target_item.node_id, edge.target_port_id))
        edge.source_item.remove_edge(edge)
        edge.target_item.remove_edge(edge)
        self.removeItem(edge)
        return True

    def _remove_annotation(self, annotation: QGraphicsObject) -> bool:
        if annotation not in self._annotations:
            return False
        self._annotations.remove(annotation)
        self.removeItem(annotation)
        return True

    def mouseMoveEvent(self, event) -> None:
        if self._pending_output is not None:
            self.drag_connection_to(event.scenePos())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            transform = self.views()[0].transform() if self.views() else QTransform()
            item = self.itemAt(event.scenePos(), transform)
            if isinstance(item, WorkflowNodeItem) and item.isSelected():
                if item._delete_button_rect().contains(item.mapFromScene(event.scenePos())):
                    self.delete_node(item)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pending_output is not None:
            success = self.finish_connection_drag_at(event.scenePos())
            event.accept()
            if success:
                return
        super().mouseReleaseEvent(event)

    def port_ref_at(self, scene_pos: QPointF, direction: str) -> WorkflowPortRef | None:
        nearest_ref: WorkflowPortRef | None = None
        nearest_distance: float | None = None

        for node in self._nodes.values():
            ports = node.widget_definition.input_ports if direction == "input" else node.widget_definition.output_ports
            for index, port in enumerate(ports):
                center = node.port_scene_position(direction, port.id)
                dx = scene_pos.x() - center.x()
                dy = scene_pos.y() - center.y()
                distance_sq = (dx * dx) + (dy * dy)
                if distance_sq > self.PORT_SCENE_HIT_RADIUS ** 2:
                    continue
                if nearest_distance is None or distance_sq < nearest_distance:
                    nearest_distance = distance_sq
                    nearest_ref = WorkflowPortRef(node.node_id, node.widget_definition.id, direction, port.id)

        return nearest_ref

    def _annotation_snapshot(self, annotation: QGraphicsObject) -> dict[str, object]:
        if isinstance(annotation, WorkflowTextAnnotationItem):
            return {
                "type": "text",
                "text": annotation.toPlainText(),
                "x": float(annotation.pos().x()),
                "y": float(annotation.pos().y()),
                "width": annotation.annotation_size().width(),
                "height": annotation.annotation_size().height(),
            }
        if isinstance(annotation, WorkflowArrowAnnotationItem):
            delta = annotation.delta
            return {
                "type": "arrow",
                "x": float(annotation.pos().x()),
                "y": float(annotation.pos().y()),
                "dx": float(delta.x()),
                "dy": float(delta.y()),
            }
        raise TypeError("Unknown annotation item")

    def _restore_annotation(self, annotation_data: dict[str, object], offset: QPointF | None = None) -> None:
        offset = offset or QPointF(0, 0)
        annotation_type = str(annotation_data.get("type", "text"))
        position = QPointF(float(annotation_data.get("x", 0.0)) + offset.x(), float(annotation_data.get("y", 0.0)) + offset.y())
        if annotation_type == "arrow":
            self.add_arrow_annotation(
                position=position,
                delta=QPointF(float(annotation_data.get("dx", 140.0)), float(annotation_data.get("dy", 0.0))),
            )
            return
        self.add_text_annotation(
            position=position,
            text=str(annotation_data.get("text", "Text annotation")),
            size=QSize(int(annotation_data.get("width", 220)), int(annotation_data.get("height", 92))),
        )

    def _begin_batch(self) -> None:
        self._batch_depth += 1

    def _end_batch(self) -> None:
        self._batch_depth = max(0, self._batch_depth - 1)

    def _notify_workflow_changed(self) -> None:
        if self._batch_depth == 0:
            self.workflowChanged.emit()


class WorkflowCanvas(QGraphicsView):
    nodeActivated = Signal(str)
    nodeDropped = Signal(str)
    statusMessage = Signal(str)
    zoomChanged = Signal(float)
    viewChanged = Signal()

    def __init__(self, widget_index: dict[str, WidgetDefinition], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = WorkflowScene(widget_index, self)
        self._scene.nodeActivated.connect(self.nodeActivated)
        self._scene.statusMessage.connect(self.statusMessage)
        self._frozen = False
        self._zoom_factor = 1.0
        self._pan_mode = False
        self._smooth_pan_target: QPointF | None = None
        self._smooth_pan_timer = QTimer(self)
        self._smooth_pan_timer.setInterval(16)
        self._smooth_pan_timer.timeout.connect(self._tick_smooth_pan)
        self._pressed_pan_keys: set[int] = set()
        self._keyboard_pan_timer = QTimer(self)
        self._keyboard_pan_timer.setInterval(16)
        self._keyboard_pan_timer.timeout.connect(self._tick_keyboard_pan)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setBackgroundBrush(QColor("#f8f8f6"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#f8f8f6"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        spacing = 34
        dot_color = QColor("#ddd9d1")
        left = int(rect.left()) - (int(rect.left()) % spacing) - spacing
        top = int(rect.top()) - (int(rect.top()) % spacing) - spacing
        right = int(rect.right()) + spacing
        bottom = int(rect.bottom()) + spacing

        painter.setPen(Qt.PenStyle.NoPen)
        for x in range(left, right + spacing, spacing):
            for y in range(top, bottom + spacing, spacing):
                painter.setBrush(dot_color)
                painter.drawEllipse(QPointF(float(x), float(y)), 1.35, 1.35)

    @property
    def workflow_scene(self) -> WorkflowScene:
        return self._scene

    def viewport_scene_center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def add_workflow_node(self, widget_id: str, position: QPointF | None = None) -> WorkflowNodeRecord:
        if position is None:
            center = self.viewport_scene_center()
            jitter = QPointF((self._scene.node_count() % 4) * 18.0, (self._scene.node_count() % 3) * 14.0)
            position = center - QPointF(54.0, 29.0) + jitter
        record = self._scene.add_node(widget_id, position)
        self.nodeDropped.emit(record.node_id)
        self.viewChanged.emit()
        return record

    def update_widget_index(self, widget_index: dict[str, WidgetDefinition]) -> None:
        self._scene.update_widget_index(widget_index)
        self.viewport().update()

    def add_text_annotation(self) -> WorkflowTextAnnotationItem:
        center = self.viewport_scene_center()
        return self._scene.add_text_annotation(position=center - QPointF(110.0, 46.0))

    def add_arrow_annotation(self) -> WorkflowArrowAnnotationItem:
        center = self.viewport_scene_center()
        return self._scene.add_arrow_annotation(position=center)

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = frozen
        self.setInteractive(not frozen)
        if frozen:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if self._pan_mode else QGraphicsView.DragMode.RubberBandDrag)
        if frozen:
            self._scene.clearSelection()

    def is_frozen(self) -> bool:
        return self._frozen

    def set_pan_mode(self, enabled: bool) -> None:
        self._pan_mode = enabled
        if self._frozen:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if enabled else QGraphicsView.DragMode.RubberBandDrag)

    def pan_mode(self) -> bool:
        return self._pan_mode

    def zoom_in(self) -> None:
        self.scale(1.15, 1.15)
        self._zoom_factor *= 1.15
        self.zoomChanged.emit(self._zoom_factor)
        self.viewChanged.emit()

    def zoom_out(self) -> None:
        self.scale(1 / 1.15, 1 / 1.15)
        self._zoom_factor /= 1.15
        self.zoomChanged.emit(self._zoom_factor)
        self.viewChanged.emit()

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)
        self.viewChanged.emit()

    def zoom_percentage(self) -> int:
        return round(self._zoom_factor * 100)

    def pan_view(self, dx: float, dy: float) -> None:
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(center + QPointF(dx, dy))
        self.viewChanged.emit()

    def smooth_pan_by(self, dx: float, dy: float) -> None:
        base = self._smooth_pan_target if self._smooth_pan_target is not None else self.viewport_scene_center()
        self._smooth_pan_target = base + QPointF(dx, dy)
        if not self._smooth_pan_timer.isActive():
            self._smooth_pan_timer.start()

    def _tick_smooth_pan(self) -> None:
        if self._smooth_pan_target is None:
            self._smooth_pan_timer.stop()
            return
        current = self.viewport_scene_center()
        delta = self._smooth_pan_target - current
        if abs(delta.x()) < 1.0 and abs(delta.y()) < 1.0:
            self.centerOn(self._smooth_pan_target)
            self._smooth_pan_target = None
            self._smooth_pan_timer.stop()
            self.viewChanged.emit()
            return
        next_center = current + QPointF(delta.x() * 0.28, delta.y() * 0.28)
        self.centerOn(next_center)
        self.viewChanged.emit()

    def _pan_vector_map(self) -> dict[int, QPointF]:
        return {
            Qt.Key.Key_Left: QPointF(-1, 0),
            Qt.Key.Key_A: QPointF(-1, 0),
            Qt.Key.Key_Right: QPointF(1, 0),
            Qt.Key.Key_D: QPointF(1, 0),
            Qt.Key.Key_Up: QPointF(0, -1),
            Qt.Key.Key_W: QPointF(0, -1),
            Qt.Key.Key_Down: QPointF(0, 1),
            Qt.Key.Key_S: QPointF(0, 1),
        }

    def _tick_keyboard_pan(self) -> None:
        if not self._pressed_pan_keys:
            self._keyboard_pan_timer.stop()
            return
        dx = 0.0
        dy = 0.0
        for key in self._pressed_pan_keys:
            vector = self._pan_vector_map().get(key)
            if vector is None:
                continue
            dx += vector.x()
            dy += vector.y()
        if dx == 0.0 and dy == 0.0:
            return
        magnitude = max((dx * dx + dy * dy) ** 0.5, 1.0)
        step = 18.0
        self.pan_view((dx / magnitude) * step, (dy / magnitude) * step)

    def export_svg(self, path: str) -> None:
        from PySide6.QtSvg import QSvgGenerator

        source_rect = self.scene().itemsBoundingRect()
        if source_rect.isNull():
            source_rect = self.sceneRect()
        source_rect = source_rect.adjusted(-32, -32, 32, 32)
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(QSize(max(1, int(source_rect.width())), max(1, int(source_rect.height()))))
        generator.setViewBox(source_rect)
        generator.setTitle("Portakal Workflow")
        painter = QPainter(generator)
        self.scene().render(painter, target=QRectF(), source=source_rect)
        painter.end()

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            self.zoom_in()
            event.accept()
            return
        if event.angleDelta().y() < 0:
            self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._frozen:
            scene_pos = self.mapToScene(event.position().toPoint())
            for node in list(self._scene._nodes.values()):
                if not node.isSelected():
                    continue
                if node._delete_button_rect().contains(node.mapFromScene(scene_pos)):
                    self._scene.delete_node(node)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if self._frozen:
            event.ignore()
            return
        if event.mimeData().hasFormat("application/x-portakal-widget"):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._frozen:
            event.ignore()
            return
        if event.mimeData().hasFormat("application/x-portakal-widget"):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if self._frozen:
            event.ignore()
            return
        if event.mimeData().hasFormat("application/x-portakal-widget"):
            widget_id = bytes(event.mimeData().data("application/x-portakal-widget")).decode("utf-8")
            scene_position = self.mapToScene(event.position().toPoint())
            self.add_workflow_node(widget_id, scene_position)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if dx or dy:
            self.viewChanged.emit()

    def keyPressEvent(self, event) -> None:
        focus_item = self._scene.focusItem()
        if isinstance(focus_item, QGraphicsTextItem) or any(
            isinstance(annotation, WorkflowTextAnnotationItem) and annotation.is_editing()
            for annotation in self._scene._annotations
        ):
            super().keyPressEvent(event)
            return
        if self._frozen and event.key() in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}:
            event.accept()
            return
        if event.key() in {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}:
            if self._scene.delete_selected_items():
                event.accept()
                return
        pan_map = self._pan_vector_map()
        if event.key() in pan_map:
            self._pressed_pan_keys.add(event.key())
            if not self._keyboard_pan_timer.isActive():
                self._keyboard_pan_timer.start()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() in self._pan_vector_map():
            self._pressed_pan_keys.discard(event.key())
            if not self._pressed_pan_keys:
                self._keyboard_pan_timer.stop()
            event.accept()
            return
        super().keyReleaseEvent(event)
