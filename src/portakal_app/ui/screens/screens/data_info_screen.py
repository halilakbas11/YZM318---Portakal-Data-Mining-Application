from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import PortakalDataError
from portakal_app.data.models import AnalysisSuggestion, DatasetHandle, DatasetSummary
from portakal_app.data.services.data_info_service import DataInfoService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.llm_analyzer import LLMAnalyzer
from portakal_app.data.services.llm_context_builder import LLMContextBuilder
from portakal_app.models import DataInfoViewModel, LLMSessionConfig, MetricCardData
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class _AnalysisWorker(QObject):
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)

    def __init__(
        self,
        token: int,
        analyzer: LLMAnalyzer,
        summary: DatasetSummary,
        context: str,
        config: LLMSessionConfig,
    ) -> None:
        super().__init__()
        self._token = token
        self._analyzer = analyzer
        self._summary = summary
        self._context = context
        self._config = config

    def run(self) -> None:
        try:
            suggestions = self._analyzer.analyze(self._summary, self._context, self._config)
        except Exception as exc:
            self.failed.emit(self._token, str(exc) or "AI analysis failed.")
        else:
            self.succeeded.emit(self._token, suggestions)
        finally:
            self.finished.emit(self._token)


class DataInfoScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._data_info_service = DataInfoService()
        self._llm_context_builder = LLMContextBuilder()
        self._llm_analyzer = LLMAnalyzer()
        self._llm_session_config = LLMSessionConfig()

        self._dataset_handle: DatasetHandle | None = None
        self._summary: DatasetSummary | None = None
        self._view_model = DataInfoViewModel()
        self._analysis_thread: QThread | None = None
        self._analysis_worker: _AnalysisWorker | None = None
        self._analysis_token = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._dataset_panel = self._build_panel("Dataset")
        dataset_layout = self._dataset_panel.layout()
        self._dataset_label = self._make_body_label("Dataset: none")
        dataset_layout.addWidget(self._dataset_label)
        layout.addWidget(self._dataset_panel)

        self._summary_panel = self._build_panel("Summary")
        self._summary_cards_layout = QGridLayout()
        self._summary_cards_layout.setHorizontalSpacing(10)
        self._summary_cards_layout.setVerticalSpacing(10)
        self._summary_panel.layout().addLayout(self._summary_cards_layout)
        layout.addWidget(self._summary_panel)

        self._columns_panel = self._build_panel("Column Profiles")
        self._column_profiles_table = QTableWidget(0, 7, self)
        self._column_profiles_table.setHorizontalHeaderLabels(
            ["Column", "Type", "Role", "Null %", "Unique", "Samples", "Summary"]
        )
        self._column_profiles_table.verticalHeader().setVisible(False)
        self._column_profiles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._column_profiles_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._column_profiles_table.horizontalHeader().setStretchLastSection(True)
        self._column_profiles_table.setMinimumHeight(220)
        self._columns_panel.layout().addWidget(self._column_profiles_table)
        layout.addWidget(self._columns_panel)

        self._llm_panel = self._build_panel("AI Analysis")
        llm_layout = self._llm_panel.layout()

        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        self._provider_label = self._make_body_label("")
        self._provider_label.setProperty("muted", True)
        header_row.addWidget(self._provider_label, 1)

        self._analyze_button = QPushButton("Analyze")
        self._analyze_button.setProperty("primary", True)
        self._analyze_button.clicked.connect(self._start_analysis)
        header_row.addWidget(self._analyze_button)
        llm_layout.addLayout(header_row)

        self._llm_status = self._make_body_label("Not analyzed yet")
        llm_layout.addWidget(self._llm_status)

        self._llm_error_label = self._make_body_label("")
        self._llm_error_label.setStyleSheet("color: #b42318; background: transparent;")
        self._llm_error_label.hide()
        llm_layout.addWidget(self._llm_error_label)

        risks_title = QLabel("Risks")
        risks_title.setProperty("sectionTitle", True)
        risks_title.setStyleSheet("font-size: 11pt; background: transparent;")
        llm_layout.addWidget(risks_title)

        self._risk_list = QListWidget(self)
        llm_layout.addWidget(self._risk_list)

        suggestions_title = QLabel("Suggestions")
        suggestions_title.setProperty("sectionTitle", True)
        suggestions_title.setStyleSheet("font-size: 11pt; background: transparent;")
        llm_layout.addWidget(suggestions_title)

        self._suggestion_list = QListWidget(self)
        llm_layout.addWidget(self._suggestion_list)
        layout.addWidget(self._llm_panel, 1)

        self._refresh_provider_label()
        self._render_view_model(DataInfoViewModel())
        self.set_dataset(None)

    def _build_panel(self, title: str) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        panel_layout = QVBoxLayout(frame)
        panel_layout.setContentsMargins(14, 12, 14, 12)
        panel_layout.setSpacing(8)
        heading = QLabel(title)
        heading.setProperty("sectionTitle", True)
        heading.setStyleSheet("font-size: 12pt; background: transparent;")
        panel_layout.addWidget(heading)
        return frame

    def _make_body_label(self, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("background: transparent;")
        return label

    def set_data_info_service(self, service: DataInfoService) -> None:
        self._data_info_service = service

    def set_llm_context_builder(self, builder: LLMContextBuilder) -> None:
        self._llm_context_builder = builder

    def set_llm_analyzer(self, analyzer: LLMAnalyzer) -> None:
        self._llm_analyzer = analyzer

    def set_llm_session_config(self, config: LLMSessionConfig) -> None:
        self._llm_session_config = config
        self._refresh_provider_label()
        self._update_analyze_button_state()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        self._analysis_token += 1
        resolved_dataset = self._resolve_dataset(dataset_handle)
        self._dataset_handle = resolved_dataset
        self._summary = None

        if resolved_dataset is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._render_view_model(
                DataInfoViewModel(
                    summary_cards=[],
                    column_profiles=[],
                    risks=[],
                    suggestions=[],
                    llm_status=i18n.t("No dataset loaded"),
                    llm_error="",
                    is_analyzing=False,
                )
            )
            self._update_analyze_button_state()
            return

        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=resolved_dataset.source.path.name))
        self._summary = self._data_info_service.summarize(resolved_dataset)
        self._render_view_model(self._data_info_service.build_from_summary(self._summary))
        self._update_analyze_button_state()

    def _resolve_dataset(self, dataset_handle: DatasetHandle | str | None) -> DatasetHandle | None:
        if isinstance(dataset_handle, DatasetHandle):
            return dataset_handle
        if isinstance(dataset_handle, str):
            try:
                return self._import_service.load(dataset_handle)
            except PortakalDataError:
                return None
        return None

    def set_view_model(self, data_info_view_model: DataInfoViewModel | None) -> None:
        self._render_view_model(data_info_view_model or DataInfoViewModel())

    def _render_view_model(self, view_model: DataInfoViewModel) -> None:
        self._view_model = view_model
        self._render_summary_cards(view_model.summary_cards)
        self._render_column_profiles(view_model)
        self._llm_status.setText(view_model.llm_status)

        if view_model.llm_error:
            self._llm_error_label.setText(view_model.llm_error)
            self._llm_error_label.show()
        else:
            self._llm_error_label.hide()

        self._populate_suggestion_list(self._risk_list, view_model.risks, empty_text="No AI risks yet.")
        self._populate_suggestion_list(
            self._suggestion_list,
            view_model.suggestions,
            empty_text="No AI suggestions yet.",
        )
        self._analyze_button.setText(i18n.t("Analyzing...") if view_model.is_analyzing else i18n.t("Analyze"))
        self._update_analyze_button_state()

    def _render_summary_cards(self, cards: list[MetricCardData]) -> None:
        while self._summary_cards_layout.count():
            item = self._summary_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not cards:
            placeholder = self._build_card(
                MetricCardData(i18n.t("Summary"), i18n.t("No dataset"), i18n.t("Load data to inspect it"))
            )
            self._summary_cards_layout.addWidget(placeholder, 0, 0)
            return

        for index, card in enumerate(cards):
            row = index // 2
            column = index % 2
            self._summary_cards_layout.addWidget(self._build_card(card), row, column)

    def _build_card(self, card: MetricCardData) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("infoCard", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        title = QLabel(card.title)
        title.setProperty("muted", True)
        title.setStyleSheet("font-size: 9pt; background: transparent;")
        layout.addWidget(title)

        value = QLabel(card.value)
        value.setProperty("sectionTitle", True)
        value.setStyleSheet("font-size: 18pt; background: transparent;")
        layout.addWidget(value)

        subtitle = QLabel(card.subtitle)
        subtitle.setWordWrap(True)
        subtitle.setProperty("muted", True)
        subtitle.setStyleSheet("font-size: 9pt; background: transparent;")
        layout.addWidget(subtitle)
        return frame

    def _render_column_profiles(self, view_model: DataInfoViewModel) -> None:
        profiles = view_model.column_profiles
        self._column_profiles_table.setRowCount(len(profiles))
        for row_index, profile in enumerate(profiles):
            sample_values = ", ".join(profile.sample_values) if profile.sample_values else "-"
            cells = [
                profile.column_name,
                profile.logical_type,
                profile.role,
                f"{profile.null_ratio * 100:.1f}%",
                str(profile.unique_count_hint),
                sample_values,
                profile.summary,
            ]
            for column_index, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._column_profiles_table.setItem(row_index, column_index, item)
        self._column_profiles_table.resizeColumnsToContents()

    def _populate_suggestion_list(
        self,
        widget: QListWidget,
        items: list[AnalysisSuggestion],
        *,
        empty_text: str,
    ) -> None:
        widget.clear()
        if not items:
            widget.addItem(QListWidgetItem(i18n.t(empty_text)))
            return
        for item in items:
            text = f"[{item.severity.upper()}] {item.title}\n{item.body}"
            widget.addItem(QListWidgetItem(text))

    def _start_analysis(self) -> None:
        if self._dataset_handle is None or self._summary is None:
            self._render_view_model(replace(self._view_model, llm_error=i18n.t("Load a dataset before running AI analysis.")))
            return
        if self._analysis_thread is not None:
            return

        self._analysis_token += 1
        token = self._analysis_token
        self._render_view_model(
            replace(
                self._view_model,
                risks=[],
                suggestions=[],
                llm_status=i18n.t("AI analysis in progress..."),
                llm_error="",
                is_analyzing=True,
            )
        )

        context = self._llm_context_builder.build(self._dataset_handle, self._summary)
        config = self._llm_session_config

        thread = QThread(self)
        worker = _AnalysisWorker(token, self._llm_analyzer, self._summary, context, config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._handle_analysis_success)
        worker.failed.connect(self._handle_analysis_failure)
        worker.finished.connect(self._handle_analysis_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.start()
        self._update_analyze_button_state()

    def _handle_analysis_success(self, token: int, suggestions: object) -> None:
        if token != self._analysis_token:
            return
        parsed_suggestions = suggestions if isinstance(suggestions, list) else []
        risks = [item for item in parsed_suggestions if isinstance(item, AnalysisSuggestion) and item.kind == "risk"]
        actions = [
            item for item in parsed_suggestions if isinstance(item, AnalysisSuggestion) and item.kind == "suggestion"
        ]
        model_text = self._llm_session_config.model.strip() or i18n.t("not set")
        self._render_view_model(
            replace(
                self._view_model,
                risks=risks,
                suggestions=actions,
                llm_status=i18n.tf(
                    "Analyzed with {provider} ({model})",
                    provider=self._llm_session_config.provider,
                    model=model_text,
                ),
                llm_error="",
                is_analyzing=False,
            )
        )

    def _handle_analysis_failure(self, token: int, error_message: str) -> None:
        if token != self._analysis_token:
            return
        self._render_view_model(
            replace(
                self._view_model,
                risks=[],
                suggestions=[],
                llm_status=i18n.t("AI analysis failed"),
                llm_error=error_message or i18n.t("AI analysis failed."),
                is_analyzing=False,
            )
        )

    def _handle_analysis_finished(self, _token: int) -> None:
        self._analysis_thread = None
        self._analysis_worker = None
        self._update_analyze_button_state()

    def _refresh_provider_label(self) -> None:
        provider = self._llm_session_config.provider
        model = self._llm_session_config.model.strip() or self._llm_session_config.model_placeholder() or i18n.t("not set")
        self._provider_label.setText(i18n.tf("Provider: {provider} | Model: {model}", provider=provider, model=model))

    def _update_analyze_button_state(self) -> None:
        self._analyze_button.setEnabled(self._dataset_handle is not None and self._analysis_thread is None)

    def help_text(self) -> str:
        return (
            "Inspect deterministic dataset profiling details and run provider-backed AI analysis manually. "
            "The AI panel shows risks and next-step suggestions while summary cards stay available even if AI fails."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/data-info/"

    def footer_status_text(self) -> str:
        if self._dataset_handle is None:
            return i18n.t("Info")
        return str(self._dataset_handle.column_count)

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.source.path.name))
        self._refresh_provider_label()
        self._render_view_model(self._view_model)
