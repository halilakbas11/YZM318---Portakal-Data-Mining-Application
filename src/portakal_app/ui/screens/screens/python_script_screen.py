from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.python_script_service import PythonScriptService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n

import re

_DEFAULT_CODE = """\
# ── Available variables ──────────────────────
#   in_data   - input data (Polars DataFrame or Orange Table*)
#   pl        - polars module
#   np/numpy  - numpy module
#   pd/pandas - pandas module
#   Orange    - Orange module (if installed)
#
# * If your code uses in_data.X, in_data.domain etc.,
#   in_data is automatically converted to an Orange Table.
#
# Assign result to out_data (Polars, Pandas, or Orange Table):
#   out_data = in_data.filter(pl.col("x") > 0)     # Polars
#   out_data = Orange.data.Table(domain, X, Y, M)   # Orange

out_data = in_data
"""

# Keyword lists for syntax highlighting
_KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
]

_BUILTINS = [
    "print", "len", "range", "list", "dict", "set", "tuple", "int",
    "float", "str", "bool", "type", "isinstance", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "abs", "round", "min", "max",
    "sum", "any", "all", "open", "input", "super", "property",
]


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules: list[tuple[re.Pattern, QTextCharFormat]] = []

        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#CF8E6D"))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        kw_pattern = r"\b(" + "|".join(_KEYWORDS) + r")\b"
        self._rules.append((re.compile(kw_pattern), kw_fmt))

        # Builtins
        bi_fmt = QTextCharFormat()
        bi_fmt.setForeground(QColor("#56A8F5"))
        bi_pattern = r"\b(" + "|".join(_BUILTINS) + r")\b"
        self._rules.append((re.compile(bi_pattern), bi_fmt))

        # Strings (single and double quote)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#6AAB73"))
        self._rules.append((re.compile(r'\"\"\".*?\"\"\"', re.DOTALL), str_fmt))
        self._rules.append((re.compile(r"\'\'\'.*?\'\'\'", re.DOTALL), str_fmt))
        self._rules.append((re.compile(r'\"[^\"]*\"'), str_fmt))
        self._rules.append((re.compile(r"\'[^\']*\'"), str_fmt))

        # Numbers
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#2AACB8"))
        self._rules.append((re.compile(r"\b\d+(\.\d+)?\b"), num_fmt))

        # Comments
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#7A7E85"))
        comment_fmt.setFontItalic(True)
        self._rules.append((re.compile(r"#[^\n]*"), comment_fmt))

        # Decorators
        dec_fmt = QTextCharFormat()
        dec_fmt.setForeground(QColor("#BBB529"))
        self._rules.append((re.compile(r"@\w+"), dec_fmt))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class PythonScriptScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = PythonScriptService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._library: list[dict[str, str]] = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top-level horizontal splitter (Left sidebar | Right editor+console)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Sidebar ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.setSpacing(6)

        # Dataset info
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        left_layout.addWidget(self._dataset_label)
        
        # Data info
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #666; font-size: 8pt;")
        left_layout.addWidget(self._info_label)

        # Library
        lib_group = QGroupBox(i18n.t("Library"))
        lib_layout = QVBoxLayout(lib_group)
        lib_layout.setContentsMargins(6, 6, 6, 6)
        self._library_list = QListWidget()
        self._library_list.currentRowChanged.connect(self._on_library_selected)
        lib_layout.addWidget(self._library_list, 1)

        lib_btns = QHBoxLayout()
        self._btn_add = QPushButton("+")
        self._btn_add.setFixedWidth(28)
        self._btn_add.clicked.connect(self._add_to_library)
        lib_btns.addWidget(self._btn_add)
        
        self._btn_remove = QPushButton("–")
        self._btn_remove.setFixedWidth(28)
        self._btn_remove.clicked.connect(self._remove_from_library)
        lib_btns.addWidget(self._btn_remove)
        
        self._btn_update = QPushButton(i18n.t("Update"))
        self._btn_update.clicked.connect(self._update_library_item)
        lib_btns.addWidget(self._btn_update)
        
        lib_btns.addStretch(1)
        lib_layout.addLayout(lib_btns)
        left_layout.addWidget(lib_group, 1)

        h_splitter.addWidget(left_panel)

        # --- Right Panel (Editor + Console) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 8, 8, 8)
        right_layout.setSpacing(0)

        # Editor header
        editor_header = QHBoxLayout()
        editor_lbl = QLabel(i18n.t("Editor"))
        editor_lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
        editor_header.addWidget(editor_lbl)
        editor_header.addStretch(1)
        right_layout.addLayout(editor_header)

        # Vertical splitter for Editor / Console
        v_splitter = QSplitter(Qt.Orientation.Vertical)

        # Code editor
        self._code_edit = QPlainTextEdit()
        self._code_edit.setPlainText(_DEFAULT_CODE)
        mono_font = QFont("Consolas", 10)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self._code_edit.setFont(mono_font)
        self._code_edit.setTabStopDistance(32.0)
        self._code_edit.setStyleSheet(
            "QPlainTextEdit { background: #2b2b2b; color: #a9b7c6; "
            "border: 1px solid #ccc; selection-background-color: #214283; }"
        )
        self._code_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = PythonHighlighter(self._code_edit.document())
        v_splitter.addWidget(self._code_edit)

        # Console header + console
        console_wrapper = QWidget()
        cw_layout = QVBoxLayout(console_wrapper)
        cw_layout.setContentsMargins(0, 4, 0, 0)
        cw_layout.setSpacing(2)

        console_lbl = QLabel(i18n.t("Console"))
        console_lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
        cw_layout.addWidget(console_lbl)

        self._console_text = QPlainTextEdit()
        self._console_text.setReadOnly(True)
        self._console_text.setFont(mono_font)
        self._console_text.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }"
        )
        self._console_text.setPlainText(">>> ")
        cw_layout.addWidget(self._console_text, 1)
        v_splitter.addWidget(console_wrapper)

        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(v_splitter, 1)

        # Return signature hint
        ret_label = QLabel("return out_data")
        ret_label.setStyleSheet("color: #888; font-family: Consolas; font-size: 9pt; padding: 4px 0;")
        right_layout.addWidget(ret_label)

        h_splitter.addWidget(right_panel)
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(h_splitter, 1)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(8, 4, 8, 8)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        bottom_bar.addWidget(self._result_label, 1)

        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        bottom_bar.addWidget(self.cb_apply_auto)

        self._run_button = QPushButton(i18n.t("Run"))
        self._run_button.setProperty("primary", True)
        self._run_button.clicked.connect(self._apply)
        self._run_button.setFixedWidth(100)
        bottom_bar.addWidget(self._run_button)
        main_layout.addLayout(bottom_bar)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            # Build data type info
            lines = []
            for col in dataset.domain.columns:
                t = col.logical_type
                if t == "numeric":
                    t = "Num"
                elif t == "categorical":
                    t = "Cat"
                elif t == "text":
                    t = "Txt"
                elif t == "datetime":
                    t = "Time"
                lines.append(f"  {col.name} ({t})")
            info_text = i18n.tf("{rows} rows × {cols} cols", rows=dataset.row_count, cols=dataset.column_count) + "\n"
            info_text += "\n".join(lines[:15])
            if len(lines) > 15:
                info_text += "\n  " + i18n.tf("... +{count} more", count=len(lines)-15)
            self._info_label.setText(info_text)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._info_label.setText("")
            self._result_label.setText("")
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "code": self._code_edit.toPlainText(),
            "library": self._library,
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        code = payload.get("code", "")
        if code:
            self._code_edit.setPlainText(str(code))
        lib = payload.get("library", [])
        if isinstance(lib, list):
            self._library = lib
            self._rebuild_library_list()

    def help_text(self) -> str:
        return "Write and execute custom Python/Polars code to transform the input data."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/pythonscript/"

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
        if self._output_dataset is not None:
            self._result_label.setText(
                i18n.tf("Output: {rows}r × {cols}c", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
            )

    def _rebuild_library_list(self) -> None:
        self._library_list.blockSignals(True)
        self._library_list.clear()
        for item in self._library:
            name = item.get("name", "Untitled")
            self._library_list.addItem(QListWidgetItem(name))
        self._library_list.blockSignals(False)

    def _on_library_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._library):
            return
        code = self._library[row].get("code", "")
        self._code_edit.setPlainText(code)

    def _add_to_library(self) -> None:
        code = self._code_edit.toPlainText()
        name = i18n.tf("Script {num}", num=len(self._library) + 1)
        self._library.append({"name": name, "code": code})
        self._rebuild_library_list()
        self._library_list.setCurrentRow(len(self._library) - 1)

    def _remove_from_library(self) -> None:
        row = self._library_list.currentRow()
        if row >= 0 and row < len(self._library):
            del self._library[row]
            self._rebuild_library_list()
            # Re-select a valid row so consecutive removals keep working
            if self._library:
                new_row = min(row, len(self._library) - 1)
                self._library_list.setCurrentRow(new_row)
            else:
                self._code_edit.setPlainText(_DEFAULT_CODE)

    def _update_library_item(self) -> None:
        row = self._library_list.currentRow()
        if row >= 0 and row < len(self._library):
            self._library[row]["code"] = self._code_edit.toPlainText()

    def _apply(self) -> None:
        code = self._code_edit.toPlainText()
        self._console_text.setPlainText(">>> Running script...\n")

        # Force UI update so user sees "Running..." immediately
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        result = self._service.execute(self._dataset_handle, code=code)

        # ── Build console output ──────────────────────────────────
        lines: list[str] = [">>> # Script executed"]

        if result.stdout:
            lines.append(result.stdout)

        if result.error:
            lines.append("")
            lines.append("─── ERROR ───")
            lines.append(result.error)

        if result.output_dataset:
            ds = result.output_dataset
            lines.append("")
            lines.append(f"✓ out_data: {ds.row_count} rows × {ds.column_count} cols")
        elif not result.error:
            lines.append("")
            lines.append("(no out_data assigned)")

        lines.append("")
        lines.append(">>> ")

        self._console_text.setPlainText("\n".join(lines))

        # Scroll to bottom
        cursor = self._console_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._console_text.setTextCursor(cursor)

        self._output_dataset = result.output_dataset

        if result.error:
            # Show first line of error in status bar
            first_line = result.error.strip().split("\n")[-1]
            self._result_label.setText(f"Error: {first_line}")
            self._result_label.setStyleSheet("color: #c75000;")
        elif self._output_dataset:
            self._result_label.setText(
                i18n.tf(
                    "Output: {rows}r × {cols}c",
                    rows=self._output_dataset.row_count,
                    cols=self._output_dataset.column_count,
                )
            )
            self._result_label.setStyleSheet("")
        else:
            self._result_label.setText(i18n.t("No output data."))
            self._result_label.setStyleSheet("")

        self._notify_output_changed()
