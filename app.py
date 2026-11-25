from __future__ import annotations

import pandas as pd
import streamlit as st

from scam_ai import (
    load_dataset, save_dataset, train_model, predict_text,
    transcribe_audio, MODEL_PATH, DATA_PATH, ensure_demo_dataset
)

st.set_page_config(page_title="Scam Detection AI Lab", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1350px;}
[data-testid="stMetric"] {background: rgba(120,130,150,.08); border: 1px solid rgba(128,128,128,.18); padding: 14px 16px; border-radius: 14px;}
div[data-testid="stTabs"] button {font-weight: 600;}
.hero {padding: 22px 24px; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; background: linear-gradient(120deg, rgba(27,83,125,.16), rgba(30,130,110,.08));}
.small-muted {color: var(--text-color); opacity: .7; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

ensure_demo_dataset()

df_now = load_dataset()
counts_now = df_now["label"].value_counts().to_dict()

with st.sidebar:
    st.markdown("## 🛡️ ScamGuard Lab")
    st.caption("Private local prototype")
    st.divider()
    st.metric("Записей в датасете", len(df_now))
    st.metric("Scam / Legitimate", f"{counts_now.get('scam', 0)} / {counts_now.get('legitimate', 0)}")
    st.metric("Файл модели", "Найден" if MODEL_PATH.exists() else "Не создан")
    st.divider()
    st.caption("Локальные файлы: data/dialogues.csv и models/scam_model.joblib")

st.markdown('<div class="hero"><h1 style="margin:0">Scam Detection AI Lab</h1><p style="margin:.45rem 0 0 0">Лаборатория анализа мошеннических диалогов. Проверяй текст, загружай аудио, записывай голос и пополняй собственный датасет.</p></div>', unsafe_allow_html=True)
st.write("")

home, audio_tab, data_tab, train_tab, info_tab = st.tabs([
    "🔎 Анализ текста", "🎙️ Аудио-тест", "📚 Датасет", "🧠 Обучение", "⚙️ О проекте"
])


def show_analysis(text: str, key_prefix: str = "result"):
    try:
        result = predict_text(text)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Threat Score", f"{result['score']}/100")
    col2.metric("Категория", result["label"])
    col3.metric("Оценка классификатора", f"{result['model_score']:.1f}/100")
    st.progress(result["score"] / 100)
    if result["signals"]:
        st.markdown("**Обнаруженные сигналы**")
        for signal in result["signals"]:
            st.write(f"- {signal}")
    else:
        st.info("Встроенные правила не нашли явных сигналов. Это не гарантирует безопасность разговора.")
    st.warning("Экспериментальная оценка, а не доказательство мошенничества и не вероятность преступления. Не принимай решение только по этому результату.")

with home:
    st.subheader("Проверка текстового диалога")
    st.write("Вставь целый разговор или его фрагмент. Чем больше контекста, тем полезнее анализ.")
    text_input = st.text_area(
        "Текст разговора", height=230, key="manual_dialogue",
        placeholder="Звонящий: Добрый день, служба безопасности банка...\nКлиент: Что произошло?\nЗвонящий: Назовите код из SMS..."
    )
    analyze_col, clear_col = st.columns([1, 5])
    with analyze_col:
        analyze = st.button("Проверить диалог", type="primary", disabled=not text_input.strip(), use_container_width=True)
    with clear_col:
        st.caption("Поддерживаются русский и казахский текст. Текущий классификатор экспериментальный.")
    if analyze:
        show_analysis(text_input)

with audio_tab:
    st.subheader("Протестировать реальный звук")
    st.write("Выбери готовый аудиофайл или запиши разговор через микрофон. После распознавания сможешь отредактировать текст и проверить его.")
    upload_col, record_col = st.columns(2, gap="large")
    uploaded = None
    recorded = None
    with upload_col:
        st.markdown("### 📁 Загрузить файл")
        uploaded = st.file_uploader("WAV, MP3, M4A, OGG или FLAC", type=["wav", "mp3", "m4a", "ogg", "flac"], key="audio_upload")
        if uploaded:
            st.audio(uploaded.getvalue(), format=uploaded.type or "audio/wav")
    with record_col:
        st.markdown("### 🎤 Записать с микрофона")
        recorded = st.audio_input("Нажми, чтобы записать голос", sample_rate=16000, key="mic_recording")
        if recorded:
            st.audio(recorded.getvalue(), format="audio/wav")

    st.divider()
    settings1, settings2, settings3 = st.columns([1, 1, 1])
    with settings1:
        model_size = st.selectbox("Модель распознавания", ["tiny", "base", "small"], index=1,
                                  help="tiny быстрее, base — компромисс, small обычно точнее, но медленнее.")
    with settings2:
        language_choice = st.selectbox("Язык речи", ["Автоопределение", "Русский", "Қазақша", "English"], index=0)
    with settings3:
        st.caption("Распознавание работает на CPU локально. Первый запуск скачает веса Whisper.")

    audio_source = uploaded if uploaded is not None else recorded
    audio_name = getattr(audio_source, "name", "recording.wav") if audio_source is not None else "recording.wav"
    language_map = {"Автоопределение": None, "Русский": "ru", "Қазақша": "kk", "English": "en"}

    if st.button("Распознать аудио", type="primary", disabled=audio_source is None, use_container_width=False):
        with st.spinner("Распознаю речь. На слабом CPU это может занять некоторое время..."):
            try:
                result = transcribe_audio(audio_source.getvalue(), audio_name, model_size, language_map[language_choice])
                st.session_state["audio_transcript"] = result["text"]
                st.session_state["audio_transcript_editor"] = result["text"]
                st.session_state["audio_detected_language"] = result["language"]
                st.success(f"Распознавание завершено. Определённый язык: {result['language']}")
            except Exception as exc:
                st.error(f"Не удалось распознать аудио: {exc}")
                st.info("Проверь установку requirements-audio.txt и подключение к интернету для первой загрузки весов Whisper.")

    if "audio_transcript_editor" not in st.session_state:
        st.session_state["audio_transcript_editor"] = st.session_state.get("audio_transcript", "")
    transcript = st.text_area("Распознанный текст (можно исправить перед анализом)",
                              height=220, key="audio_transcript_editor")
    if transcript.strip():
        a1, a2 = st.columns([1, 3])
        with a1:
            analyze_audio = st.button("Анализировать текст", type="primary", key="analyze_transcript")
        with a2:
            label_for_audio = st.selectbox("Метка для датасета", ["scam", "legitimate"], key="audio_training_label")
        if analyze_audio:
            show_analysis(transcript, "audio")
        if st.button("Сохранить транскрипт в датасет", key="save_transcript"):
            df = load_dataset()
            entry = pd.DataFrame([{"text": transcript.strip(), "label": label_for_audio, "source": "audio_transcription"}])
            merged = pd.concat([df, entry], ignore_index=True).drop_duplicates(subset=["text", "label"])
            save_dataset(merged)
            st.success("Текст добавлен в датасет. Для использования в модели её нужно переобучить.")
    st.caption("Важно: приложение не сохраняет аудиофайлы на диск по умолчанию. В датасет добавляется только текст по твоему отдельному нажатию.")

with data_tab:
    st.subheader("Датасет обучения")
    df = load_dataset()
    counts = df["label"].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    c1.metric("Всего примеров", len(df))
    c2.metric("Мошеннические", counts.get("scam", 0))
    c3.metric("Обычные", counts.get("legitimate", 0))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Добавить текстовый пример")
    with st.form("add_text_example", clear_on_submit=True):
        sample_text = st.text_area("Диалог", height=130)
        sample_label = st.selectbox("Правильная метка", ["scam", "legitimate"])
        sample_source = st.text_input("Источник / заметка", value="manual")
        submitted = st.form_submit_button("Добавить пример")
    if submitted:
        if len(sample_text.strip()) < 15:
            st.error("Добавь более содержательный диалог (минимум 15 символов).")
        else:
            df.loc[len(df)] = [sample_text.strip(), sample_label, sample_source.strip() or "manual"]
            save_dataset(df.drop_duplicates(subset=["text", "label"]))
            st.success("Пример сохранён. Переобучи модель во вкладке «Обучение».")

    st.markdown("### Импортировать набор CSV")
    st.caption("Файл должен содержать колонки text и label; значения label: scam или legitimate.")
    csv_file = st.file_uploader("Выбрать CSV", type=["csv"], key="dataset_csv")
    if csv_file is not None:
        try:
            imported = pd.read_csv(csv_file).fillna("")
            if not {"text", "label"}.issubset(imported.columns):
                st.error("В CSV отсутствуют колонки text и label.")
            else:
                imported = imported[["text", "label"]].copy()
                imported["source"] = "csv_import"
                imported["label"] = imported["label"].astype(str).str.lower().str.strip()
                imported = imported[imported["label"].isin(["scam", "legitimate"])]
                st.write(f"Корректных строк для импорта: {len(imported)}")
                if st.button("Добавить строки из CSV", key="merge_csv"):
                    merged = pd.concat([df, imported], ignore_index=True).drop_duplicates(subset=["text", "label"])
                    save_dataset(merged)
                    st.success(f"Всего строк после импорта: {len(merged)}")
        except Exception as exc:
            st.error(f"Ошибка чтения CSV: {exc}")

    st.download_button("Скачать датасет CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="scam_dialogues.csv", mime="text/csv")

with train_tab:
    st.subheader("Обучение классификатора")
    df = load_dataset()
    counts = df["label"].value_counts().to_dict()
    x1, x2, x3 = st.columns(3)
    x1.metric("Train data", len(df))
    x2.metric("Scam", counts.get("scam", 0))
    x3.metric("Legitimate", counts.get("legitimate", 0))
    st.markdown("Модель использует словесные и символьные признаки TF-IDF и Logistic Regression. Это лёгкий baseline, не большая языковая модель.")
    st.warning("Стартовый датасет содержит искусственно составленные примеры. Высокие метрики на нём не означают, что модель надёжно распознаёт реальные звонки.")
    if st.button("Обучить / переобучить модель", type="primary"):
        with st.spinner("Обучаю модель и проверяю её на отложенной части датасета..."):
            try:
                metrics = train_model(df)
                st.success("Модель сохранена в models/scam_model.joblib")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                m2.metric("Precision", f"{metrics['precision']:.1%}")
                m3.metric("Recall", f"{metrics['recall']:.1%}")
                m4.metric("F1", f"{metrics['f1']:.1%}")
                st.caption(f"Обучение: {metrics['train_size']} примеров. Тест: {metrics['test_size']} примеров. Случайное разделение, random_state=42.")
                cm = metrics["confusion_matrix"]
                st.markdown("**Матрица ошибок** (строки — настоящий класс, столбцы — прогноз; порядок: legitimate, scam)")
                st.dataframe(pd.DataFrame(cm, index=["Actual legitimate", "Actual scam"],
                                          columns=["Predicted legitimate", "Predicted scam"]), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
    if MODEL_PATH.exists():
        st.success("Обученная модель найдена локально.")
    else:
        st.info("Модель ещё не обучена. Нажми кнопку выше.")

with info_tab:
    st.subheader("Что это за версия")
    st.markdown("""
**В этой сборке уже есть:**
- анализ текстовых диалогов;
- загрузка WAV / MP3 / M4A / OGG / FLAC;
- запись голоса с микрофона прямо в браузере;
- распознавание речи с помощью multilingual Whisper;
- ручное исправление транскрипта;
- сохранение размеченного текста в локальный датасет;
- обучение и проверка baseline-классификатора;
- поддержка первых русских и казахских демонстрационных примеров.

**Пока нет:**
- прямого доступа к аудио обычного SIM-звонка на Android;
- надёжного разделения голосов на «оператор / клиент»;
- полноценной казахской модели, обученной на большом корпусе;
- потокового анализа телефонного разговора;
- гарантированно точного определения мошенничества.
    """)
    st.info("Не обрабатывай частные разговоры без законного основания и необходимого уведомления/согласия участников. Перед добавлением в датасет удаляй ФИО, телефоны, адреса, номера карт, пароли и одноразовые коды.")
    st.caption("Приложение рассчитано на локальный запуск. Не публикуй папки data и models с личными данными или приватными примерами.")
