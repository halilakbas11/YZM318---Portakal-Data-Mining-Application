from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class I18nManager(QObject):
    language_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._lang = "en"
        self._translations: dict[str, dict[str, str]] = {
            "tr": {
                "Language / Dil": "Dil",
                "Turkish": "Türkçe",
                "English": "İngilizce",
                "Documentation": "Dokümantasyon",
                "Settings": "Ayarlar",
                "Help": "Yardım",
                "File": "Dosya",
                "Edit": "Düzen",
                "View": "Görünüm",
                "Widget": "Bileşen",
                "Window": "Pencere",
                "Options": "Seçenekler",
                "New": "Yeni",
                "Open": "Aç",
                "Open and Freeze": "Aç ve Sabitle",
                "Open Recent": "Son Açılanlar",
                "Open Workflow": "İş Akışını Aç",
                "Open Workflow and Freeze": "İş Akışını Aç ve Sabitle",
                "Open Source": "Kaynağı Aç",
                "Close": "Kapat",
                "Close Window": "Pencereyi Kapat",
                "Quit": "Çıkış",
                "Remove": "Kaldır",
                "Rename": "Yeniden Adlandır",
                "Data": "Veri",
                "Transform": "Dönüştür",
                "CSV File Import": "CSV Dosya İçe Aktarma",
                "Datasets": "Veri Setleri",
                "Data Table": "Veri Tablosu",
                "Paint Data": "Veri Boyama",
                "Data Info": "Veri Bilgisi",
                "Rank": "Sırala",
                "Edit Domain": "Alanı Düzenle",
                "Color": "Renk",
                "Column Statistics": "Sütun İstatistikleri",
                "Select Columns": "Sütun Seç",
                "Normalize": "Normalleştir",
                "Scatter Plot": "Dağılım Grafiği",
                "Linear Regression": "Doğrusal Regresyon",
                "Test & Score": "Test ve Puanla",
                "Visualize": "Görselleştir",
                "Model": "Model",
                "Evaluate": "Değerlendir",
                "Unsupervised": "Gözetimsiz",
                "PCA": "PCA",
                "Reload Last Workflow": "Son İş Akışını Yeniden Yükle",
                "File:": "Dosya:",
                "URL:": "URL:",
                "File Type": "Dosya Türü",
                "Select a local dataset...": "Yerel bir veri seti seçin...",
                "Paste a remote dataset URL...": "Uzak veri seti URL'sini yapıştırın...",
                "Determine type from the file extension": "Türü dosya uzantısından belirle",
                "Basket file (*.basket *.bsk)": "Basket dosyası (*.basket *.bsk)",
                "Comma-separated values (*.csv *.csv.gz *.gz *.csv.bz2 *.bz2 *.csv.xz *.xz)": "Virgülle ayrılmış değerler (*.csv *.csv.gz *.gz *.csv.bz2 *.bz2 *.csv.xz *.xz)",
                "Microsoft Excel 97-2004 spreadsheet (*.xls)": "Microsoft Excel 97-2004 çalışma sayfası (*.xls)",
                "Microsoft Excel spreadsheet (*.xlsx)": "Microsoft Excel çalışma sayfası (*.xlsx)",
                "Pickled Orange data (*.pkl *.pickle *.pkl.gz *.pickle.gz *.gz *.pkl.bz2 *.pickle.bz2 *.bz2 *.pkl.xz *.pickle.xz *.xz)": "Pickle Orange verisi (*.pkl *.pickle *.pkl.gz *.pickle.gz *.gz *.pkl.bz2 *.pickle.bz2 *.bz2 *.pkl.xz *.pickle.xz *.xz)",
                "Tab-separated values (*.tab *.tsv *.tab.gz *.tsv.gz *.gz *.tab.bz2 *.tsv.bz2 *.bz2 *.tab.xz *.tsv.xz *.xz)": "Sekmeyle ayrılmış değerler (*.tab *.tsv *.tab.gz *.tsv.gz *.gz *.tab.bz2 *.tsv.bz2 *.bz2 *.tab.xz *.tsv.xz *.xz)",
                "Filter widgets...": "Bileşenleri filtrele...",
                "Coming soon": "Yakında",
                "Send Automatically": "Otomatik Gönder",
                "Apply Automatically": "Otomatik Uygula",
                "Apply Domain": "Alanı Uygula",
                "Apply Import": "Veriyi Uygula",
                "Apply": "Uygula",
                "Reset": "Sıfırla",
                "Reload": "Yeniden Yükle",
                "Source": "Kaynak",
                "Info": "Bilgi",
                "Summary": "Özet",
                "Summary card": "Özet kartı",
                "Column Profiles": "Sütun Profilleri",
                "AI Analysis": "YZ Analizi",
                "Columns (Double click to edit)": "Sütunlar (Düzenlemek için çift tıkla)",
                "No dataset selected": "Veri seti seçilmedi",
                "Choose a local file or URL to inspect metadata.": "Meta verileri incelemek için dosya veya URL seçin.",
                "Error loading URL": "URL yüklenirken hata oluştu",
                "Apply failed": "Uygulama başarısız oldu",
                "Kaggle notebook/code URLs are not supported. Use a dataset URL in the form https://www.kaggle.com/datasets/<owner>/<dataset-name>.": "Kaggle notebook/code bağlantıları desteklenmiyor. https://www.kaggle.com/datasets/<owner>/<dataset-name> biçiminde bir veri seti bağlantısı kullanın.",
                "Only Kaggle dataset URLs are supported. Use a URL in the form https://www.kaggle.com/datasets/<owner>/<dataset-name>.": "Yalnızca Kaggle veri seti bağlantıları destekleniyor. https://www.kaggle.com/datasets/<owner>/<dataset-name> biçiminde bir bağlantı kullanın.",
                "Select a file path or URL before applying.": "Uygulamadan önce dosya yolu veya URL seçin.",
                "Local file source": "Yerel dosya kaynağı",
                "Source path": "Kaynak yolu",
                "Detected extension": "Algılanan uzantı",
                "columns discovered": "sütun bulundu",
                "rows detected": "satır bulundu",
                "Cache file": "Önbellek dosyası",
                "unknown": "bilinmiyor",
                "No source selected yet.": "Henüz kaynak seçilmedi.",
                "Use Browse, Reload and Apply after the data backend is connected.": "Veri altyapısı bağlandıktan sonra Gözat, Yeniden Yükle ve Uygula kullanın.",
                "Name": "Ad",
                "Type": "Tür",
                "Role": "Rol",
                "Values": "Değerler",
                "Kaggle User:": "Kaggle Kullanıcı:",
                "Kaggle API Key:": "Kaggle API Anahtarı:",
                "Optional: your Kaggle username": "İsteğe bağlı: Kaggle kullanıcı adınız",
                "Optional: paste Kaggle API key": "İsteğe bağlı: Kaggle API anahtarınızı yapıştırın",
                "Variable X:": "Değişken X:",
                "Variable Y:": "Değişken Y:",
                "Add Label": "Etiket Ekle",
                "Remove Tool": "Silme Aracı",
                "Send Data": "Veriyi Gönder",
                "Refresh Ranking": "Sıralamayı Yenile",
                "Feature Ranking": "Özellik Sıralaması",
                "Controls": "Kontroller",
                "Target": "Hedef",
                "Feature Filter": "Özellik Filtresi",
                "Top N": "En İyi N",
                "Dataset: none": "Veri Seti: Yok",
                "Dataset: {name}": "Veri Seti: {name}",
                "Analyze": "Analiz Et",
                "Analyzing...": "Analiz Ediliyor...",
                "Risks": "Riskler",
                "Suggestions": "Öneriler",
                "Save Data": "Veriyi Kaydet",
                "Save": "Kaydet",
                "Load": "Yükle",
                "Save as ...": "Farklı Kaydet ...",
                "Save As ...": "Farklı Kaydet ...",
                "Save Workflow": "İş Akışını Kaydet",
                "Save Workflow Image as SVG ...": "İş Akışı Görselini SVG Olarak Kaydet ...",
                "Export Workflow SVG": "İş Akışı SVG Dışa Aktar",
                "Autosave when receiving new data": "Yeni veri geldiğinde otomatik kaydet",
                "Add type annotations to header": "Başlığa tür açıklamaları ekle",
                "Delimited Source": "Ayrılmış Kaynak",
                "Import Options": "İçe Aktarma Seçenekleri",
                "Preview": "Önizleme",
                "Workflow": "İş Akışı",
                "Drag widgets from the catalog onto the canvas. Click a node to inspect it. Build connections by dragging from an output port onto a compatible input port.": "Bileşenleri katalogdan tuvale sürükleyin. İncelemek için bir düğüme tıklayın. Bir çıkış portundan uyumlu bir giriş portuna sürükleyerek bağlantı kurun.",
                "Search for data set ...": "Veri seti ara ...",
                "First parsed row is header": "İlk satır başlık olarak kullanılsın",
                "No imported dataset": "İçe aktarılan veri yok",
                "Choose a delimited file and preview it before importing.": "İçe aktarmadan önce ayrılmış dosya seçin ve önizleyin.",
                "Import options changed. Preview or apply to refresh the parsed dataset.": "İçe aktarma seçenekleri değişti. Ayrıştırılmış veriyi güncellemek için önizleme yapın veya uygulayın.",
                "Select a delimited file first.": "Önce ayrılmış bir dosya seçin.",
                "Current workflow dataset loaded. Import options are not available for this source.": "Mevcut iş akışı verisi yüklendi. Bu kaynak için içe aktarma seçenekleri kullanılamaz.",
                "Dataset": "Veri Seti",
                "Load a dataset to edit column names, types and roles.": "Sütun adlarını, türlerini ve rollerini düzenlemek için bir veri seti yükleyin.",
                "Columns": "Sütunlar",
                "Restore Inferred": "Tahmin Edileni Geri Yükle",
                "Domain changes applied to the workflow dataset.": "Alan değişiklikleri iş akışı verisine uygulandı.",
                "Restored inferred domain from the current dataset.": "Mevcut veri setinden tahmin edilen alan geri yüklendi.",
                "Load a dataset to rank feature usefulness.": "Özelliklerin önemini sıralamak için bir veri seti yükleyin.",
                "No target selected. Ranking uses heuristic mode.": "Hedef seçilmedi. Sıralama sezgisel modda yapılıyor.",
                "Search": "Ara",
                "Column": "Sütun",
                "Statistics": "İstatistikler",
                "Distribution": "Dağılım",
                "Top Values": "En Sık Değerler",
                "Load a dataset to inspect per-column statistics.": "Sütun bazlı istatistikleri görmek için bir veri seti yükleyin.",
                "No columns matched the current search.": "Mevcut aramaya uyan sütun bulunamadı.",
                "No warnings": "Uyarı yok",
                "Download Preview": "Önizlemeyi İndir",
                "Description": "Açıklama",
                "Select a dataset to review its description and source.": "Açıklamasını ve kaynağını görmek için bir veri seti seçin.",
                "No datasets match the current filter.": "Mevcut filtreyle eşleşen veri seti yok.",
                "Domain:": "Alan:",
                "No dataset loaded.": "Veri seti yüklenmedi.",
                "No data loaded yet": "Henüz veri yüklenmedi",
                "No dataset": "Veri yok",
                "Load data to inspect it": "İncelemek için veri yükleyin",
                "Loading data...": "Veriler yükleniyor...",
                "Polars DataFrame is being prepared, please wait.": "Polars DataFrame hazırlanıyor, lütfen bekleyin.",
                "You can start by selecting a file from the File widget.": "File bileşeninden bir dosya seçerek başlayabilirsiniz.",
                "Variables": "Değişkenler",
                "Selection": "Seçim",
                "Show variable labels (if present)": "Değişken etiketlerini göster (varsa)",
                "Visualize numeric values": "Sayısal değerleri görselleştir",
                "Color by instance classes": "Sınıf etiketine göre renklendir",
                "Clear Selection": "Seçimi Temizle",
                "Select full rows": "Tam satır seç",
                "Restore Original Order": "Orijinal Sırayı Geri Yükle",
                "Data: none": "Veri: yok",
                "Data Subset: -": "Veri Alt Kümesi: -",
                "Selected": "Seçili",
                "Points": "Noktalar",
                "Selected: {count}": "Seçili: {count}",
                "Points: {count}": "Noktalar: {count}",
                "Names": "Adlar",
                "Source Columns": "Kaynak Sütunları",
                "Output Names": "Çıktı Adları",
                "X Column:": "X Sütunu:",
                "Y Column:": "Y Sütunu:",
                "Label Column:": "Etiket Sütunu:",
                "Labels": "Etiketler",
                "None": "Yok",
                "Tools": "Araçlar",
                "Brush": "Fırça",
                "Put": "Ekle",
                "Select": "Seç",
                "Jitter": "Titreşim",
                "Magnet": "Mıknatıs",
                "Clear": "Temizle",
                "Radius:": "Yarıçap:",
                "Intensity:": "Yoğunluk:",
                "Symbol:": "Sembol:",
                "Reset to Input Data": "Girdi Verisine Sıfırla",
                "Discrete Variables": "Ayrık Değişkenler",
                "Numeric Variables": "Sayısal Değişkenler",
                "No discrete variables available for manual color assignment.": "Elle renk ataması yapılabilecek ayrık değişken yok.",
                "No numeric variables available for gradient palettes.": "Renk geçişi uygulanabilecek sayısal değişken yok.",
                "Save Color Settings": "Renk Ayarlarını Kaydet",
                "Load Color Settings": "Renk Ayarlarını Yükle",
                "Select Color": "Renk Seç",
                "Sample Values": "Örnek Değerler",
                "No AI risks yet.": "Henüz YZ riski yok.",
                "No AI suggestions yet.": "Henüz YZ önerisi yok.",
                "No dataset loaded": "Veri seti yüklenmedi",
                "Load a dataset before running AI analysis.": "YZ analizini çalıştırmadan önce bir veri seti yükleyin.",
                "AI analysis in progress...": "YZ analizi sürüyor...",
                "AI analysis failed": "YZ analizi başarısız oldu",
                "AI analysis failed.": "YZ analizi başarısız oldu.",
                "not set": "ayarlanmadı",
                "Provider": "Sağlayıcı",
                "Provider: {provider} | Model: {model}": "Sağlayıcı: {provider} | Model: {model}",
                "Analyzed with {provider} ({model})": "{provider} ile analiz edildi ({model})",
                "Editing domain for {count} columns. Apply changes to update the workflow dataset.": "{count} sütun için alan düzenleniyor. İş akışı verisini güncellemek için değişiklikleri uygulayın.",
                "{target_label} ({count} values)": "{target_label} ({count} değer)",
                "{count} instances ({missing_summary})": "{count} gözlem ({missing_summary})",
                "{count} missing values": "{count} eksik değer",
                "{count} numeric features": "{count} sayısal özellik",
                "Target with {count} values": "{count} değerli hedef",
                "No meta attributes.": "Meta öznitelik yok.",
                "Data: {dataset_name}: {row_count} instances, {column_count} variables": "Veri: {dataset_name}: {row_count} gözlem, {column_count} değişken",
                "Selected Data: {dataset_name}: {row_count} instances, {column_count} variables": "Seçili Veri: {dataset_name}: {row_count} gözlem, {column_count} değişken",
                "Selected Data: {row_count} instances, {column_count} variables": "Seçili Veri: {row_count} gözlem, {column_count} değişken",
                "Features: {count} numeric ({missing_summary})": "Özellikler: {count} sayısal ({missing_summary})",
                "Target: {target_label}": "Hedef: {target_label}",
                "Showing all rows in the table": "Tablodaki tüm satırlar gösteriliyor",
                "no missing values": "eksik değer yok",
                "contains missing values": "eksik değer içeriyor",
                "no missing data": "eksik veri yok",
                "none": "yok",
                "Duplicate": "Çoğalt",
                "Copy": "Kopyala",
                "Paste": "Yapıştır",
                "Select all": "Tümünü Seç",
                "Load a file dataset.": "Bir dosya veri seti yükleyin.",
                "Preset-based import flow.": "Hazır ayarlara dayalı içe aktarma akışı.",
                "Explore packaged datasets.": "Paketli veri setlerini keşfedin.",
                "Inspect loaded rows.": "Yüklenen satırları inceleyin.",
                "Edit data manually.": "Veriyi elle düzenleyin.",
                "Profile the dataset.": "Veri setinin profilini çıkarın.",
                "Rank features.": "Özellikleri sıralayın.",
                "Manage column roles.": "Sütun rollerini yönetin.",
                "Assign color metadata.": "Renk meta verilerini atayın.",
                "Deep dive into column distributions.": "Sütun dağılımlarını ayrıntılı inceleyin.",
                "Export the current dataset.": "Geçerli veri setini dışa aktarın.",
                "Choose feature sets.": "Özellik kümelerini seçin.",
                "Scale numeric columns.": "Sayısal sütunları ölçekleyin.",
                "Explore points visually.": "Noktaları görsel olarak inceleyin.",
                "Train a baseline model.": "Temel bir model eğitin.",
                "Measure model performance.": "Model performansını ölçün.",
                "Reduce dimensionality.": "Boyutu azaltın.",
                "Transform widgets are planned but not part of this shell milestone.": "Dönüştürme bileşenleri planlandı ancak bu kabuk kilometre taşına dahil değil.",
                "Visualization widgets will be integrated by a later group.": "Görselleştirme bileşenleri daha sonraki bir grup tarafından entegre edilecek.",
                "Model widgets will be integrated by a later group.": "Model bileşenleri daha sonraki bir grup tarafından entegre edilecek.",
                "Evaluation widgets will be integrated by a later group.": "Değerlendirme bileşenleri daha sonraki bir grup tarafından entegre edilecek.",
                "Unsupervised widgets will be integrated by a later group.": "Gözetimsiz öğrenme bileşenleri daha sonraki bir grup tarafından entegre edilecek.",
                "Text Annotation": "Metin Notu",
                "Arrow Annotation": "Ok Notu",
                "Window Groups": "Pencere Grupları",
                "Expand Tool Dock": "Araç Bölmesini Genişlet",
                "Log": "Günlük",
                "Zoom in": "Yakınlaştır",
                "Zoom out": "Uzaklaştır",
                "Reset Zoom": "Yakınlaştırmayı Sıfırla",
                "Show Workflow Margins": "İş Akışı Kenar Boşluklarını Göster",
                "Bring Widgets to Front": "Bileşenleri Öne Getir",
                "Display Widgets on Top": "Bileşenleri Üstte Göster",
                "Workflow Info": "İş Akışı Bilgisi",
                "Title": "Başlık",
                "Cancel": "İptal",
                "Open widget menu": "Bileşen menüsünü aç",
                "Open data preview": "Veri önizlemesini aç",
                "Open selected data preview": "Seçili veri önizlemesini aç",
                "Open Dataset...": "Veri Seti Aç...",
                "Reload Source": "Kaynağı Yeniden Yükle",
                "Data Preview": "Veri Önizleme",
                "Selected Data Preview": "Seçili Veri Önizleme",
                "Status: {status}": "Durum: {status}",
                "Center": "Ortala",
                "Bring to Front": "Öne Getir",
                "Always on Top": "Her Zaman Üstte",
                "Widget Help": "Bileşen Yardımı",
                "No help available.": "Yardım içeriği yok.",
                "No preview available.": "Önizleme yok.",
                "Report": "Rapor",
                "Write a comment...": "Bir yorum yazın...",
                "Back to Last Schema": "Son Şemaya Dön",
                "Save Report": "Raporu Kaydet",
                "Comments": "Yorumlar",
                "Pan canvas": "Tuvali kaydır",
                "Reset zoom": "Yakınlaştırmayı sıfırla",
                "Add text annotation": "Metin notu ekle",
                "Add arrow annotation": "Ok notu ekle",
                "Show more tools": "Daha fazla araç göster",
                "Hide extra tools": "Ek araçları gizle",
                "Reset Widget Settings...": "Bileşen Ayarlarını Sıfırla...",
                "Add-ons...": "Eklentiler...",
                "About Portakal": "Portakal Hakkında",
                "Untitled": "Adsız",
                "Language changed. Open widgets were updated immediately.": "Dil değiştirildi. Açık bileşenler anında güncellendi.",

                # ── Connection dialog ──
                "Edit Links": "Bağlantıları Düzenle",

                # ── Select by Data Index ──
                "Data Subset": "Veri Alt Kümesi",
                "Matching Data": "Eşleşen Veri",
                "Non-matching Data": "Eşleşmeyen Veri",

                # ── Randomize ──
                "Shuffled columns": "Karıştırılan sütunlar",
                "Classes": "Sınıflar",
                "Features": "Özellikler",
                "Metas": "Meta Veriler",
                "Shuffled rows": "Karıştırılan satırlar",
                "All": "Tümü",
                "Replicable shuffling": "Tekrarlanabilir karıştırma",
                "Randomization successful.": "Rastgeleleştirme başarılı.",

                # ── Purge Domain ──
                "Sort categorical feature values": "Kategorik özellik değerlerini sırala",
                "Remove unused feature values": "Kullanılmayan özellik değerlerini kaldır",
                "Remove constant features": "Sabit özellikleri kaldır",
                "Sorted: -, reduced: -, removed: -": "Sıralanan: -, azaltılan: -, kaldırılan: -",
                "Sort categorical class values": "Kategorik sınıf değerlerini sırala",
                "Remove unused class variable values": "Kullanılmayan sınıf değişkeni değerlerini kaldır",
                "Remove constant class variables": "Sabit sınıf değişkenlerini kaldır",
                "Meta attributes": "Meta öznitelikler",
                "Remove unused meta attribute values": "Kullanılmayan meta öznitelik değerlerini kaldır",
                "Remove constant meta attributes": "Sabit meta öznitelikleri kaldır",
                "Send": "Gönder",

                # ── Unique ──
                "Group By": "Gruplama",
                "Tiebreaker": "Eşitlik Kırıcı",
                "Instance to select in each group:": "Her grupta seçilecek örnek:",
                "First instance": "İlk örnek",
                "Last instance": "Son örnek",
                "Middle instance": "Ortadaki örnek",
                "Random instance": "Rastgele örnek",
                "Discard non-unique": "Tekil olmayanları çıkar",
                "Select All": "Tümünü Seç",
                "Deselect All": "Seçimi Kaldır",
                "Input: {before} → Output: {after} ({removed} duplicates removed)": "Girdi: {before} → Çıktı: {after} ({removed} kopya çıkarıldı)",

                # ── Apply Domain ──
                "Template": "Şablon",
                "Output:": "Çıktı:",
                "Data:": "Veri:",

                # ── Data Sampler ──
                "Sampling Type": "Örnekleme Türü",
                "Fixed proportion of data:": "Sabit veri oranı:",
                "Fixed sample size": "Sabit örneklem boyutu",
                "Instances:": "Örnekler:",
                "Sample with replacement": "Yerine koymalı örnekleme",
                "Cross validation": "Çapraz doğrulama",
                "Number of subsets:": "Alt küme sayısı:",
                "Unused subset:": "Kullanılmayan alt küme:",
                "Bootstrap": "Önyükleme (Bootstrap)",
                "Replicable (deterministic) sampling": "Tekrarlanabilir (belirleyici) örnekleme",
                "Stratify sample (when possible)": "Tabakalı örnekleme (mümkünse)",
                "Sample Data": "Veriyi Örnekle",
                "Sample:": "Örneklem:",
                "Remaining:": "Kalan:",
                "Target: {col}": "Hedef: {col}",
                "Input: {n} rows": "Girdi: {n} satır",
                "Sample: {n} rows": "Örneklem: {n} satır",
                "Remaining: {n} rows": "Kalan: {n} satır",

                # ── Select Columns ──
                "Ignored": "Yoksayılan",
                "Features >": "Özellikler >",
                "Target >": "Hedef >",
                "Meta >": "Meta >",
                "< Ignored": "< Yoksayılan",
                "Meta": "Meta",
                "{n} features": "{n} özellik",
                "{n} meta": "{n} meta",
                "{n} ignored (dropped)": "{n} yoksayılan (çıkarıldı)",

                # ── Select Rows ──
                "Conditions": "Koşullar",
                "+ Add Condition": "+ Koşul Ekle",
                "Remove All": "Tümünü Kaldır",
                "All conditions must match (AND)": "Tüm koşullar sağlansın (VE)",
                "At least one condition (OR)": "En az bir koşul sağlansın (VEYA)",
                "Selected: {n}": "Seçilen: {n}",
                "Remaining: {n}": "Kalan: {n}",
                "Total: {n}": "Toplam: {n}",
                "Matching:": "Eşleşen:",
                "Unmatched:": "Eşleşmeyen:",
                "Unmatched Data": "Eşleşmeyen Veri",
                "Remove unused values and constant features": "Kullanılmayan değerleri ve sabit özellikleri kaldır",
                "Remove unused classes": "Kullanılmayan sınıfları kaldır",

                # ── Transpose ──
                "Feature names": "Özellik adları",
                "Generic": "Genel",
                "Type a prefix ...": "Bir önek yazın ...",
                "From variable:": "Değişkenden:",
                "Result:": "Sonuç:",

                # ── Split ──
                "Variable": "Değişken",
                "Delimiter:": "Ayırıcı:",
                "Output Values": "Çıktı Değerleri",
                "Categorical (No, Yes)": "Kategorik (Hayır, Evet)",
                "Numerical (0, 1)": "Sayısal (0, 1)",
                "Counts": "Sayımlar",
                "Added": "Eklenen",
                "indicator column(s)": "gösterge sütunu/sütunları",

                # ── Merge Data ──
                "Append columns from Extra Data": "Ekstra Veriden sütun ekle",
                "Find matching pairs of rows": "Eşleşen satır çiftlerini bul",
                "Concatenate tables": "Tabloları birleştir",
                "Row matching": "Satır eşleştirme",
                "matches": "eşleşme",
                "Row index": "Satır indeksi",
                "MERGE": "BİRLEŞTİR",
                "Please select valid columns to match.": "Lütfen eşleştirmek için geçerli sütunlar seçin.",
                "Left Join": "Sol Birleştirme",
                "Inner Join": "İç Birleştirme",
                "Outer Join": "Dış Birleştirme",
                "Merge Error:": "Birleştirme Hatası:",
                "rows x": "satır x",
                "columns": "sütun",

                # ── Concatenate ──
                "Variable Sets Merging": "Değişken Kümelerini Birleştirme",
                "When there is no primary table, the output should contain": "Birincil tablo olmadığında, çıktı şunları içermelidir",
                "all variables that appear in input tables": "giriş tablolarındaki tüm değişkenler",
                "only variables that appear in all tables": "tüm tablolarda ortak olan değişkenler",
                "The resulting table will have a class only if there is no conflict\nbetween input classes.": "Sonuç tablosu, yalnızca giriş sınıfları arasında\nçakışma yoksa bir sınıf içerecektir.",
                "Variable matching": "Değişken eşleştirme",
                "Use column names from the primary table,\nand ignore names in other tables.": "Birincil tablodaki sütun adlarını kullan,\ndiğer tablolardaki adları yoksay.",
                "Treat variables with the same name as the same variable,\neven if they are computed using different formulae.": "Aynı ada sahip değişkenleri aynı değişken olarak kabul et,\nfarklı formüllerle hesaplansalar bile.",
                "Source Identification": "Kaynak Tanımlama",
                "Append data source IDs": "Veri kaynak kimliklerini ekle",
                "Feature name:": "Özellik adı:",
                "Place:": "Konum:",
                "Class attribute": "Sınıf özniteliği",
                "Meta attribute": "Meta öznitelik",
                "Feature": "Özellik",
                "Primary:": "Birincil:",
                "Additional:": "Ek:",
                "No output.": "Çıktı yok.",

                # ── Aggregate Columns ──
                "All": "Tümü",
                "All, including meta attributes": "Meta öznitelikler dahil tümü",
                "Selected variables": "Seçili değişkenler",
                "Aggregation": "Toplama",
                "Operation:": "İşlem:",
                "Mean": "Ortalama",
                "Output column:": "Çıktı sütunu:",
                "agg": "top",
                "Mean of": "Ortalama:",
                "column(s) ->": "sütun ->",
                "Input:": "Girdi:",
                "rows": "satır",

                # ── Group By ──
                "Attributes": "Öznitelikler",
                "Aggregations": "Toplamalar",
                "Median": "Ortanca",
                "Q1": "Ç1",
                "Q3": "Ç3",
                "Min. value": "Min. değer",
                "Max. value": "Maks. değer",
                "Mode": "Mod",
                "Standard deviation": "Standart sapma",
                "Variance": "Varyans",
                "Sum": "Toplam",
                "Concatenate": "Birleştir",
                "Span": "Aralık",
                "First value": "İlk değer",
                "Last value": "Son değer",
                "Random value": "Rastgele değer",
                "Count defined": "Tanımlı sayısı",
                "Count": "Sayı",
                "Proportion defined": "Tanımlı oranı",
                "Groups:": "Gruplar:",
                "Columns:": "Sütunlar:",

                # ── Pivot Table ──
                "Pivot Settings": "Pivot Ayarları",
                "Row:": "Satır:",
                "Column:": "Sütun:",
                "Value:": "Değer:",
                "(Count)": "(Sayı)",
                "Aggregation:": "Toplama:",
                "Pivot:": "Pivot:",

                # ── Preprocess ──
                "Preprocessors": "Ön İşlemciler",
                "Continuize Discrete Variables": "Ayrık Değişkenleri Sürekli Yap",
                "Impute Missing Values": "Eksik Değerleri Doldur",
                "Select Relevant Features": "İlgili Özellikleri Seç",
                "Select Random Features": "Rastgele Özellikleri Seç",
                "Normalize Features": "Özellikleri Normalleştir",
                "Randomize": "Rastgeleleştir",
                "Remove Sparse Features": "Seyrek Özellikleri Kaldır",
                "Principal Component Analysis": "Temel Bileşen Analizi",
                "CUR Matrix Decomposition": "CUR Matris Ayrışımı",
                "Most frequent is base": "En sık olan temeldir",
                "One feature per value": "Her değer için bir özellik",
                "Remove non-binary features": "İkili olmayan özellikleri kaldır",
                "Remove categorical features": "Kategorik özellikleri kaldır",
                "Treat as ordinal": "Sıralı olarak işle",
                "Divide by number of values": "Değer sayısına böl",
                "Average/Most frequent": "Ortalama/En sık",
                "Replace with random value": "Rastgele değerle değiştir",
                "Remove rows with missing values": "Eksik değerli satırları kaldır",
                "Standardize to μ=0, σ²=1": "μ=0, σ²=1 olarak standartlaştır",
                "Center to μ=0": "μ=0 olarak ortala",
                "Scale to σ²=1": "σ²=1 olarak ölçekle",
                "Normalize to interval [-1, 1]": "[-1, 1] aralığına normalleştir",
                "Normalize to interval [0, 1]": "[0, 1] aralığına normalleştir",
                "Score": "Puan",
                "Information Gain": "Bilgi Kazancı",
                "Gain Ratio": "Kazanç Oranı",
                "Gini Index": "Gini İndeksi",
                "ReliefF": "ReliefF",
                "ANOVA": "ANOVA",
                "Chi2": "Ki-Kare",
                "Univariate Linear Regression": "Tek Değişkenli Doğrusal Regresyon",
                "Number of features": "Özellik sayısı",
                "Fixed:": "Sabit:",
                "Proportion:": "Oran:",
                "Percentage:": "Yüzde:",
                "Meta data": "Meta veri",
                "Remove features with too many": "Çok fazla olan özellikleri kaldır",
                "missing values": "eksik değerler",
                "zeros": "sıfırlar",
                "Threshold:": "Eşik:",
                "Successfully transformed.": "Başarıyla dönüştürüldü.",
                "Before:": "Önce:",
                "After:": "Sonra:",
                "Error during preprocessing:": "Ön işleme sırasında hata:",

                # ── Impute ──
                "Default Method": "Varsayılan Yöntem",
                "Don't impute": "Doldurma yapma",
                "As a distinct value": "Ayrı bir değer olarak",
                "Fixed values": "Sabit değerler",
                "value:": "değer:",
                "Random values": "Rastgele değerler",
                "seed:": "tohum:",
                "Remove instances with unknown values": "Bilinmeyen değerli örnekleri kaldır",
                "Individual Attribute Settings": "Bireysel Öznitelik Ayarları",
                "Attribute": "Öznitelik",
                "Imputation Method": "Doldurma Yöntemi",
                "Restore All to Default": "Tümünü Varsayılana Geri Yükle",
                "(Default)": "(Varsayılan)",
                "Imputed. Remaining missing:": "Dolduruldu. Kalan eksik:",
                "Rows:": "Satırlar:",
                "Impute Failed:": "Doldurma Başarısız:",

                # ── Continuize ──
                "Categorical Variables": "Kategorik Değişkenler",
                "Continuous Variables": "Sürekli Değişkenler",
                "Keep as it is": "Olduğu gibi bırak",
                "Reset All": "Tümünü Sıfırla",
                "Error:": "Hata:",

                # ── Discretize ──
                "Discretize Settings": "Ayrıklaştırma Ayarları",
                "Keep numeric": "Sayısal olarak tut",
                "Natural binning, desired bins:": "Doğal gruplama, istenen grup sayısı:",
                "Fixed width:": "Sabit genişlik:",
                "Time interval:": "Zaman aralığı:",
                "year(s)": "yıl",
                "month(s)": "ay",
                "week(s)": "hafta",
                "day(s)": "gün",
                "hour(s)": "saat",
                "minute(s)": "dakika",
                "second(s)": "saniye",
                "Equal frequency, intervals:": "Eşit sıklık, aralık sayısı:",
                "Equal width, intervals:": "Eşit genişlik, aralık sayısı:",
                "Entropy vs. MDL": "Entropi ve MDL",
                "Custom:": "Özel:",
                "e.g. 0.0, 0.5, 1.0": "ör. 0.0, 0.5, 1.0",
                "Use default setting": "Varsayılan ayarı kullan",

                # ── Melt ──
                "Unique Row Identifier": "Benzersiz Satır Tanımlayıcı",
                "Row number": "Satır numarası",
                "Filter": "Filtre",
                "Ignore non-numeric features": "Sayısal olmayan özellikleri yoksay",
                "Exclude zero values": "Sıfır değerleri hariç tut",
                "Names for generated features": "Oluşturulan özellikler için adlar",
                "Item:": "Öğe:",
                "item": "öğe",
                "value": "değer",
                "Error applying Melt:": "Melt uygulama hatası:",

                # ── Create Class ──
                "New Class Name": "Yeni Sınıf Adı",
                "class": "sınıf",
                "Match by Substring": "Alt Dizeye Göre Eşle",
                "From column:": "Sütundan:",
                "Substring": "Alt Dize",
                "Use regular expressions": "Düzenli ifadeler kullan",
                "Match only at the beginning": "Yalnızca başlangıçta eşle",
                "Case sensitive": "Büyük/küçük harf duyarlı",
                "Error mapping class:": "Sınıf eşleme hatası:",

                # ── Create Instance ──
                "Filter...": "Filtre...",
                "Random": "Rastgele",
                "Input": "Girdi",
                "Append this instance to input data": "Bu örneği girdi verisine ekle",
                "Create": "Oluştur",
                "Created instance. Output:": "Örnek oluşturuldu. Çıktı:",
                "Error creating instance:": "Örnek oluşturma hatası:",

                # ── Formula ──
                "Variable Definitions": "Değişken Tanımları",
                "Name...": "Ad...",
                "Expression...": "İfade...",
                "Select Column": "Sütun Seç",
                "Select Function": "Fonksiyon Seç",
                "New variable": "Yeni değişken",
                "New variables": "Yeni değişkenler",
                "Applied": "Uygulandı",
                "formula(s). New columns:": "formül. Yeni sütunlar:",
                "Formula Error:": "Formül Hatası:",

                # ── Python Script ──
                "Editor": "Düzenleyici",
                "Console": "Konsol",
                "Run": "Çalıştır",
                "No output data.": "Çıktı verisi yok.",
                "Library": "Kütüphane",
                "Update": "Güncelle",
                "Script": "Betik",

                # ── Common transform strings ──
                "cols": "sütun",
                "rows ×": "satır ×",
                "(Num)": "(Say)",
                "(Cat)": "(Kat)",
                "(Txt)": "(Mtn)",
                "(Time)": "(Zaman)",

                # ── Missing widget labels (catalog) ──
                "Select by Data Index": "Veri İndeksine Göre Seç",
                "Purge Domain": "Alanı Temizle",
                "Unique": "Tekil",
                "Data Sampler": "Veri Örnekleyici",
                "Select Rows": "Satır Seç",
                "Transpose": "Transpoze",
                "Split": "Böl",
                "Merge Data": "Veriyi Birleştir",
                "Aggregate Columns": "Sütunları Topla",
                "Pivot Table": "Pivot Tablo",
                "Preprocess": "Ön İşle",
                "Impute": "Doldur",
                "Continuize": "Sürekli Yap",
                "Discretize": "Ayrıklaştır",
                "Melt": "Erit",
                "Create Class": "Sınıf Oluştur",
                "Create Instance": "Örnek Oluştur",
                "Formula": "Formül",
                "Python Script": "Python Betiği",

                # ── Missing catalog descriptions ──
                "Match rows by index subset.": "Satırları indeks alt kümesine göre eşle.",
                "Shuffle rows or columns.": "Satırları veya sütunları karıştır.",
                "Remove unused values and constant features.": "Kullanılmayan değerleri ve sabit özellikleri kaldır.",
                "Filter duplicate rows.": "Tekrarlayan satırları filtrele.",
                "Apply template domain structure.": "Şablon alan yapısını uygula.",
                "Sample a subset of the data.": "Verinin bir alt kümesini örnekle.",
                "Filter rows by conditions.": "Koşullara göre satır filtrele.",
                "Flip rows and columns.": "Satır ve sütunları çevir.",
                "Split string column into indicators.": "Metin sütununu göstergelere böl.",
                "Join two datasets by column values.": "İki veri setini sütun değerlerine göre birleştir.",
                "Append datasets vertically.": "Veri setlerini dikey olarak ekle.",
                "Compute row-wise aggregations.": "Satır bazlı toplamalar hesapla.",
                "Group and aggregate data.": "Veriyi grupla ve topla.",
                "Create cross-tabulations.": "Çapraz tablolama oluştur.",
                "Build preprocessing pipelines.": "Ön işleme boru hatları oluştur.",
                "Fill missing values.": "Eksik değerleri doldur.",
                "Convert categorical to numeric.": "Kategorik değerleri sayısala çevir.",
                "Convert numeric to categorical.": "Sayısal değerleri kategorik yapmak.",
                "Wide to long format.": "Geniş formatı uzun formata çevir.",
                "Create class from string patterns.": "Metin kalıplarından sınıf oluştur.",
                "Create a single data instance.": "Tek bir veri örneği oluştur.",
                "Construct features with expressions.": "İfadelerle özellik oluştur.",
                "Run custom Python code.": "Özel Python kodu çalıştır.",

                # ── Missing dynamic / status strings ──
                "Data: -": "Veri: -",
                "Data: {name}": "Veri: {name}",
                "Data: {rows} rows, {cols} columns": "Veri: {rows} satır, {cols} sütun",
                "Data Subset: {rows} rows": "Veri Alt Kümesi: {rows} satır",
                "Matching: -  |  Non-matching: -  |  Total: -": "Eşleşen: -  |  Eşleşmeyen: -  |  Toplam: -",
                "Matching: {m}  |  Non-matching: {nm}  |  Total: {total}": "Eşleşen: {m}  |  Eşleşmeyen: {nm}  |  Toplam: {total}",
                "{value}%": "%{value}",
                "{value} %": "%{value}",
                "Error: {err}": "Hata: {err}",
                "Error: {error}": "Hata: {error}",
                "Sorted: {sorted}, reduced: {reduced}, removed: {removed}": "Sıralanan: {sorted}, azaltılan: {reduced}, kaldırılan: {removed}",
                "Reduced: -, removed: -": "Azaltılan: -, kaldırılan: -",
                "Reduced: {reduced}, removed: {removed}": "Azaltılan: {reduced}, kaldırılan: {removed}",
                "Error applying purge.": "Temizleme uygulama hatası.",
                "Rows: {before} -> {after} ({removed} duplicates removed)": "Satırlar: {before} -> {after} ({removed} tekrar kaldırıldı)",
                "Template: -": "Şablon: -",
                "Template: {name} ({cols} columns)": "Şablon: {name} ({cols} sütun)",
                "Output: -": "Çıktı: -",
                "Output: {rows} rows, {cols} columns": "Çıktı: {rows} satır, {cols} sütun",
                "Sample: {sample} rows  |  Remaining: {remaining} rows": "Örneklem: {sample} satır  |  Kalan: {remaining} satır",
                "Matching: {m}  |  Unmatched: {nm}": "Eşleşen: {m}  |  Eşleşmeyen: {nm}",
                "Result: {after} columns (was {before})": "Sonuç: {after} sütun ({before} idi)",
                "Result: {rows} rows x {cols} columns": "Sonuç: {rows} satır x {cols} sütun",
                "Data: -  |  Extra: -": "Veri: -  |  Ekstra: -",
                "Data: {d}  |  Extra: {e}": "Veri: {d}  |  Ekstra: {e}",
                "Primary: -  |  Additional: -": "Birincil: -  |  Ek: -",
                "Primary: {p}  |  Additional: {a}": "Birincil: {p}  |  Ek: {a}",
                "{op} of {col_count} column(s) -> '{out_name}'  |  Input: {in_count} rows  |  Output: {out_count} rows": "{col_count} sütunun {op}'sı -> '{out_name}'  |  Girdi: {in_count} satır  |  Çıktı: {out_count} satır",
                "Groups: {groups} | Columns: {columns}": "Gruplar: {groups} | Sütunlar: {columns}",
                "Pivot: {rows} rows x {columns} columns": "Pivot: {rows} satır x {columns} sütun",
                "Successfully transformed.\nBefore: {before_r}r x {before_c}c  ->  After: {after_r}r x {after_c}c": "Başarıyla dönüştürüldü.\nÖnce: {before_r}s x {before_c}sü  ->  Sonra: {after_r}s x {after_c}sü",
                "Error during preprocessing: {error}": "Ön işleme sırasında hata: {error}",
                "Error mapping class: {error}": "Sınıf eşleme hatası: {error}",
                "Created '{name}' with {count} matching categories": "'{name}' {count} eşleşen kategoriyle oluşturuldu",
                "Created instance. Output: {rows} rows": "Örnek oluşturuldu. Çıktı: {rows} satır",
                "Error creating instance: {error}": "Örnek oluşturma hatası: {error}",
                "Applied {count} formula(s). New columns: {new_cols}. Output: {rows}r x {cols}c": "{count} formül uygulandı. Yeni sütunlar: {new_cols}. Çıktı: {rows}s x {cols}sü",
                "Formula Error: {error}": "Formül Hatası: {error}",
                "Script {num}": "Betik {num}",
                "Error applying Melt: {error}": "Erit uygulama hatası: {error}",
                "Added {count} indicator column(s)": "{count} gösterge sütunu eklendi",
                "Merge Error: {err}": "Birleştirme Hatası: {err}",
                "Output: {rows}r × {cols}c": "Çıktı: {rows}s × {cols}sü",
                "{rows} rows × {cols} cols": "{rows} satır × {cols} sütun",
                "... +{count} more": "... +{count} daha",
                "and {count} more": "ve {count} daha",

                # ── Help / description strings ──
                "Data rows keep their identity even when some or all original variables "
                "are replaced by variables computed from the original ones.\n\n"
                "This widget gets two data tables (\"Data\" and \"Data Subset\") that "
                "can be traced back to the same source. It selects all rows from Data "
                "that appear in Data Subset, based on row identity and not actual data.":
                "Veri satırları, orijinal değişkenlerin bazıları veya tümü orijinallerinden hesaplanan "
                "değişkenlerle değiştirilse bile kimliklerini korur.\n\n"
                "Bu bileşen, aynı kaynağa kadar izlenebilen iki veri tablosu (\"Veri\" ve \"Veri Alt Kümesi\") "
                "alır. Gerçek verilere değil, satır kimliğine göre Veri Alt Kümesi'nde görünen tüm satırları "
                "Veri'den seçer.",
                "Apply the domain (column structure, roles, and types) from a Template dataset "
                "to the Data input.":
                "Bir Şablon veri setinden alan yapısını (sütun yapısı, roller ve türler) "
                "Veri girdisine uygula.",
                "Compute a row-wise aggregation (sum, mean, etc.) over selected numeric columns.":
                "Seçilen sayısal sütunlar üzerinde satır bazlı toplama (toplam, ortalama vb.) hesapla.",
                "Group the dataset by selected columns and compute per-attribute aggregations.":
                "Veri setini seçilen sütunlara göre grupla ve öznitelik bazlı toplamalar hesapla.",
                "Create a pivot (cross-tabulation) table from the dataset.":
                "Veri setinden pivot (çapraz tablolama) tablo oluştur.",
                "Build a preprocessing pipeline: remove missing values, constant features, normalize, or standardize.":
                "Ön işleme boru hattı oluştur: eksik değerleri kaldır, sabit özellikleri sil, normalleştir veya standartlaştır.",
                "Not yet natively implemented in Portakal.": "Henüz Portakal'da yerel olarak uygulanmadı.",

                # ── Miscellaneous missing ──
                "Print": "Yazdır",
                "Selected Data: -": "Seçili Veri: -",
                "Save As...": "Farklı Kaydet...",
                "No target variable inferred": "Hedef değişken çıkarılamadı",
                "Target rank": "Hedef sıralaması",
                "Value": "Değer",
                " seed: ": " tohum: ",
                " value: ": " değer: ",
                "Remove redundant instance": "Artık örneği kaldır",
                "Instance id": "Örnek kimliği",
            }
        }
        self._reverse_translations: dict[str, dict[str, str]] = {
            lang: {translated: source for source, translated in values.items()}
            for lang, values in self._translations.items()
        }

    def set_language(self, lang: str) -> None:
        if lang not in ["en", "tr"]:
            lang = "en"
        if self._lang != lang:
            self._lang = lang
            self.language_changed.emit(lang)

    def current_language(self) -> str:
        return self._lang

    def t(self, text: str) -> str:
        if self._lang == "en":
            reverse = self._reverse_translations.get("tr", {})
            return reverse.get(text, text)
        return self._translations.get(self._lang, {}).get(text, text)

    def tf(self, text: str, **kwargs) -> str:
        return self.t(text).format(**kwargs)

    def apply_to_widget(self, root) -> None:
        try:
            from PySide6.QtGui import QAction
            from PySide6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QTableWidget, QWidget
        except Exception:
            return

        if root is None:
            return

        widgets = [root] + list(root.findChildren(QWidget))
        for widget in widgets:
            if isinstance(widget, QGroupBox):
                widget.setTitle(self.t(widget.title()))
            if isinstance(widget, QLabel):
                widget.setText(self.t(widget.text()))
            if isinstance(widget, QAbstractButton):
                widget.setText(self.t(widget.text()))
            if isinstance(widget, QLineEdit):
                placeholder = widget.placeholderText()
                if placeholder:
                    widget.setPlaceholderText(self.t(placeholder))
            if isinstance(widget, QComboBox) and widget.count() > 0:
                current = widget.currentText()
                for index in range(widget.count()):
                    item_text = widget.itemText(index)
                    widget.setItemText(index, self.t(item_text))
                if current:
                    widget.setCurrentText(self.t(current))
            if isinstance(widget, QTableWidget):
                for index in range(widget.columnCount()):
                    header = widget.horizontalHeaderItem(index)
                    if header is not None:
                        header.setText(self.t(header.text()))

        for action in root.findChildren(QAction):
            action.setText(self.t(action.text()))

# Global instance
_app_i18n = I18nManager()

def set_language(lang: str) -> None:
    _app_i18n.set_language(lang)

def current_language() -> str:
    return _app_i18n.current_language()

def t(text: str) -> str:
    return _app_i18n.t(text)


def tf(text: str, **kwargs) -> str:
    return _app_i18n.tf(text, **kwargs)

def on_language_changed(callback) -> None:
    _app_i18n.language_changed.connect(callback)


def apply_to_widget(root) -> None:
    _app_i18n.apply_to_widget(root)
