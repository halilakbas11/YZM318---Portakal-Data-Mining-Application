from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from portakal_app.ui.catalog import build_categories, build_widgets
from portakal_app.ui.screens.bag_of_words_screen import BagOfWordsScreen
from portakal_app.ui.screens.corpus_viewer_screen import CorpusViewerScreen
from portakal_app.ui.screens.corpus_screen import CorpusScreen
from portakal_app.ui.screens.create_corpus_screen import CreateCorpusScreen
from portakal_app.ui.screens.document_map_screen import DocumentMapScreen
from portakal_app.ui.screens.extract_keywords_screen import ExtractKeywordsScreen
from portakal_app.ui.screens.import_documents_screen import ImportDocumentsScreen
from portakal_app.ui.screens.preprocess_text_screen import PreprocessTextScreen
from portakal_app.ui.screens.sentiment_analysis_screen import SentimentAnalysisScreen
from portakal_app.ui.screens.text_statistics_screen import TextStatisticsScreen
from portakal_app.ui.screens.topic_modelling_screen import TopicModellingScreen
from portakal_app.ui.screens.word_cloud_screen import WordCloudScreen
from portakal_app.ui.screens.word_list_screen import WordListScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


TEXT_MINING_WIDGETS = {
    "text-corpus": ("Corpus", "text_corpus", ("Corpus",), ("Corpus",), CorpusScreen),
    "text-corpus-viewer": ("Corpus Viewer", "text_corpus_viewer", ("Corpus",), ("Corpus",), CorpusViewerScreen),
    "text-import-documents": ("Import Documents", "text_import_documents", (), ("Corpus",), ImportDocumentsScreen),
    "text-create-corpus": ("Create Corpus", "text_create_corpus", (), ("Corpus",), CreateCorpusScreen),
    "text-preprocess": ("Preprocess Text", "text_preprocess", ("Corpus",), ("Corpus",), PreprocessTextScreen),
    "text-bag-of-words": ("Bag of Words", "text_bag_of_words", ("Corpus",), ("Corpus",), BagOfWordsScreen),
    "text-statistics": ("Statistics", "text_statistics", ("Corpus",), ("Corpus",), TextStatisticsScreen),
    "text-word-list": (
        "Word List",
        "text_word_list",
        ("Corpus", "Words"),
        ("Words", "Selected Words", "Data"),
        WordListScreen,
    ),
    "text-word-cloud": (
        "Word Cloud",
        "text_word_cloud",
        ("Corpus",),
        ("Corpus", "Selected Word", "Word Counts"),
        WordCloudScreen,
    ),
    "text-extract-keywords": ("Extract Keywords", "text_extract_keywords", ("Corpus",), ("Words",), ExtractKeywordsScreen),
    "text-sentiment-analysis": (
        "Sentiment Analysis",
        "text_sentiment_analysis",
        ("Corpus",),
        ("Corpus",),
        SentimentAnalysisScreen,
    ),
    "text-topic-modelling": (
        "Topic Modelling",
        "text_topic_modelling",
        ("Corpus",),
        ("Corpus", "Topic"),
        TopicModellingScreen,
    ),
    "text-document-map": ("Document Map", "text_document_map", ("Corpus",), (), DocumentMapScreen),
}


def test_text_mining_category_and_widgets_are_registered(app):
    categories = {category.id: category for category in build_categories()}
    widgets = {widget.id: widget for widget in build_widgets()}

    assert categories["text-mining"].label == "Text Mining"
    for widget_id, (label, icon_name, inputs, outputs, screen_type) in TEXT_MINING_WIDGETS.items():
        widget = widgets[widget_id]
        assert widget.category_id == "text-mining"
        assert widget.label == label
        assert widget.icon_name == icon_name
        assert tuple(port.label for port in widget.input_ports) == inputs
        assert tuple(port.label for port in widget.output_ports) == outputs
        assert isinstance(widget.screen_factory(), screen_type)


def test_text_mining_icon_assets_exist():
    assets_dir = Path(__file__).resolve().parents[1] / "src" / "portakal_app" / "ui" / "assets"
    expected_icons = {"text_mining", *(metadata[1] for metadata in TEXT_MINING_WIDGETS.values())}

    for icon_name in expected_icons:
        assert (assets_dir / f"{icon_name}.svg").exists()


def test_create_corpus_to_preprocess_to_bag_of_words_payload_flow(app):
    create = CreateCorpusScreen()
    corpus = CorpusScreen()
    preprocess = PreprocessTextScreen()
    bag_of_words = BagOfWordsScreen()

    document = create.add_document("Workflow Text", "Apple, apple! Banana.", "Manual")
    corpus.set_input_payload(create.current_output_payload())
    preprocess.set_input_payload(corpus.current_output_payload())
    bag_of_words.set_input_payload(preprocess.current_output_payload())

    assert corpus.current_output_payload().value == (document,)
    assert preprocess.current_output_payload().port_label == "Corpus"
    assert preprocess.current_output_payload().value[0].text == "apple apple banana"
    assert bag_of_words.current_output_payload().port_label == "Corpus"
    assert bag_of_words._table.rowCount() > 0
    assert "apple" in [
        bag_of_words._table.horizontalHeaderItem(index).text()
        for index in range(bag_of_words._table.columnCount())
    ]
