from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import re
import tempfile
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dialogues.csv"
AUGMENTATION_PATH = BASE_DIR / "data" / "curated_training.csv"
MODEL_PATH = BASE_DIR / "models" / "scam_model.joblib"

# These are hand-written demo examples. They are useful for testing the app,
# but are NOT a representative corpus of real scam calls.
DEMO_ROWS = [
    ("Сотрудник банка: Назовите код из СМС, чтобы отменить подозрительную операцию. Клиент: Хорошо, сейчас прочитаю.", "scam"),
    ("Звонящий: Переведите деньги на безопасный счет, иначе их украдут. Клиент: Какой счет?", "scam"),
    ("Звонящий: Установите AnyDesk и продиктуйте код подключения, я помогу защитить деньги.", "scam"),
    ("Звонящий: Срочно сообщите пароль от мобильного банка, иначе доступ будет заблокирован.", "scam"),
    ("Звонящий: Ваш родственник попал в аварию, срочно отправьте деньги, времени нет.", "scam"),
    ("Звонящий: Вы выиграли приз, оплатите комиссию для получения выигрыша.", "scam"),
    ("Звонящий: Не кладите трубку и никому не рассказывайте, что мы проводим проверку.", "scam"),
    ("Звонящий: Назовите номер карты и CVV для возврата ошибочного платежа.", "scam"),
    ("Звонящий: Я из службы безопасности. Перейдите по ссылке и введите данные карты.", "scam"),
    ("Звонящий: Чтобы сохранить деньги, снимите наличные и передайте курьеру.", "scam"),
    ("Звонящий: Вам нужно подтвердить перевод кодом, который пришел на телефон.", "scam"),
    ("Звонящий: Если не выполните инструкции прямо сейчас, вас привлекут к ответственности.", "scam"),
    ("Звонящий: Мы обнаружили кредит на ваше имя. Установите приложение удаленного доступа.", "scam"),
    ("Звонящий: Для отмены кредита сообщите одноразовый пароль из сообщения.", "scam"),
    ("Звонящий: Переведите средства на временный счет Национального банка.", "scam"),
    ("Звонящий: Я сотрудник полиции, помогите поймать преступников и никому не говорите.", "scam"),
    ("Звонящий: Ваш аккаунт заблокирован. Отправьте фото удостоверения и данные карты.", "scam"),
    ("Звонящий: Оплатите доставку по ссылке, иначе посылка будет уничтожена.", "scam"),
    ("Звонящий: Продиктуйте код подтверждения, это не пароль, а техническая проверка.", "scam"),
    ("Звонящий: Не отключайте демонстрацию экрана, я должен видеть ваше приложение банка.", "scam"),
    ("Банк қызметкері: Күмәнді операцияны тоқтату үшін SMS арқылы келген кодты айтыңыз.", "scam"),
    ("Ақшаңызды қорғау үшін қаражатты қауіпсіз шотқа аударыңыз.", "scam"),
    ("AnyDesk орнатып, қашықтан қосылу кодын жіберіңіз.", "scam"),
    ("Картаңыз бұғатталмауы үшін PIN код пен CVV нөмірін айтыңыз.", "scam"),
    ("Сізге полициядан хабарласып тұрмын. Ешкімге айтпаңыз және ақшаны қазір аударыңыз.", "scam"),
    ("Сіздің атыңызға несие рәсімделді. Оны жою үшін бір реттік парольді айтыңыз.", "scam"),
    ("Сіз жүлде ұтып алдыңыз, ақшаны алу үшін комиссия төлеу керек.", "scam"),
    ("Телефонды қоймаңыз, нұсқауларды дәл қазір орындаңыз.", "scam"),
    ("Экранды көрсетуді қосыңыз, банк қосымшасын бірге тексереміз.", "scam"),
    ("Кодты айтпасаңыз, шотыңыз бұғатталады.", "scam"),
    ("Банк шотыңызды қорғау үшін келген SMS кодты маған оқып беріңіз.", "scam"),
    ("Қазір ақшаңызды тексеру керек, қаражатты қауіпсіз шотқа аударыңыз.", "scam"),
    ("Тексеру аяқталғанша қоңырауды үзбеңіз және отбасыңызға айтпаңыз.", "scam"),
    ("Курьер сізге келеді, қолма-қол ақшаны соған беріңіз.", "scam"),
    ("Сілтеме арқылы кіріп, банк логиніңіз бен құпиясөзіңізді енгізіңіз.", "scam"),
    ("Здравствуйте, это клиника. Напоминаем, что ваша запись на прием завтра в 10:30.", "legitimate"),
    ("Добрый день, доставка. Курьер будет у подъезда примерно через двадцать минут.", "legitimate"),
    ("Здравствуйте, это университет. Расписание занятий опубликовано в личном кабинете.", "legitimate"),
    ("Подтвердите, пожалуйста, удобно ли вам перенести встречу на пятницу?", "legitimate"),
    ("Это служба поддержки. Мы зарегистрировали ваше обращение под номером 2451.", "legitimate"),
    ("Напоминаем об оплате интернета. Счет можно посмотреть в официальном приложении.", "legitimate"),
    ("Здравствуйте, ваш заказ готов к выдаче. Забрать его можно до конца недели.", "legitimate"),
    ("Я звоню по поводу собеседования. Вам удобно поговорить несколько минут?", "legitimate"),
    ("Ваш талон к врачу на 14:00. Возьмите удостоверение личности.", "legitimate"),
    ("Мы получили вашу заявку и отправили подтверждение на электронную почту.", "legitimate"),
    ("Курьер не смог дозвониться. Вы можете выбрать другое время доставки в приложении.", "legitimate"),
    ("Оплата прошла успешно. Чек доступен в истории операций.", "legitimate"),
    ("Здравствуйте, это преподаватель. Завтра занятие начнется на полчаса позже.", "legitimate"),
    ("Мы хотим уточнить качество обслуживания после вашего обращения.", "legitimate"),
    ("Ваша карта готова. Получить ее можно в отделении с удостоверением.", "legitimate"),
    ("Спасибо за покупку. Гарантия действует двенадцать месяцев.", "legitimate"),
    ("Встреча подтверждена на вторник в 15:00. До свидания.", "legitimate"),
    ("Ваш заказ передан в доставку. Статус можно проверить на сайте магазина.", "legitimate"),
    ("Напоминаем, что срок действия абонемента заканчивается в конце месяца.", "legitimate"),
    ("Мы получили документы. Если потребуется уточнение, специалист свяжется с вами.", "legitimate"),
    ("Сәлеметсіз бе, дәрігерге жазылуыңыз ертең сағат онға расталды.", "legitimate"),
    ("Курьер жиырма минуттан кейін келеді. Үйде боласыз ба?", "legitimate"),
    ("Тапсырысыңыз жеткізу пунктіне келді, мәртебесін ресми қосымшада көре аласыз.", "legitimate"),
    ("Университеттің сабақ кестесі жеке кабинетте жарияланды.", "legitimate"),
    ("Сіздің өтінішіңіз тіркелді, маман жұмыс күні ішінде хабарласады.", "legitimate"),
    ("Интернет төлемі сәтті өтті, түбіртек қосымшада қолжетімді.", "legitimate"),
    ("Кездесуді жұма күніне ауыстыру сізге ыңғайлы ма?", "legitimate"),
    ("Құжаттарыңызды алдық, қосымша ақпарат қажет болса хабарласамыз.", "legitimate"),
    ("Сіздің картаңыз дайын, оны бөлімшеден жеке куәлікпен ала аласыз.", "legitimate"),
    ("Тапсырысыңыз жөнелтілді, жеткізу барысын ресми қосымшада тексеріңіз.", "legitimate"),
    ("Тапсырысыңызды алу үшін жеке куәлігіңізді көрсетіңіз.", "legitimate"),
    ("Қабылдауыңыз бейсенбі күні сағат он беске ауыстырылды.", "legitimate"),
    ("Өтінішіңіз бойынша жауап электрондық поштаға жіберілді.", "legitimate"),
    ("Төлем түбіртегін мобильді қосымшадан жүктей аласыз.", "legitimate"),
    ("Сізге ыңғайлы болса, кездесуді келесі аптаға белгілейік.", "legitimate"),
]

SIGNALS = {
    "Запрос SMS-кода или одноразового пароля": [
        "код из смс", "код из sms", "код подтверждения", "одноразовый пароль", "продиктуйте код", "смс-код",
        "sms арқылы келген код", "бір реттік пароль", "кодты айтыңыз", "кодты айтып", "sms кодты", "смс кодты", "смс-кодты"
    ],
    "Запрос банковских секретов": [
        "cvv", "cvc", "пароль от мобильного банка", "данные карты", "пин-код", "pin код", "құпиясөз", "құпия сөз"
    ],
    "Перевод на «безопасный» или временный счёт": [
        "безопасный счет", "безопасный счёт", "временный счет", "временный счёт", "қауіпсіз шот", "қауіпсіз шотқа"
    ],
    "Удалённый доступ к устройству": [
        "anydesk", "rustdesk", "удаленного доступа", "удалённого доступа", "демонстрацию экрана",
        "қашықтан қосылу", "экранды көрсет"
    ],
    "Давление срочностью, секретностью или угрозами": [
        "срочно", "прямо сейчас", "не кладите трубку", "никому не рассказывайте", "иначе вас", "времени нет",
        "дәл қазір", "ешкімге айтпаңыз", "телефонды қоймаңыз", "бұғатталады"
    ],
    "Подозрительный выигрыш или комиссия": [
        "выиграли приз", "оплатите комиссию", "получения выигрыша", "жүлде ұтып", "комиссия төле"
    ],
}


_DISCLOSURE_ACTION = re.compile(
    r"(?<!\w)(?:назов\w*|сообщ\w*|продикт\w*|дикту\w*|скаж\w*|прочит\w*|"
    r"переда\w*|отправ\w*|пришл\w*|перешл\w*|озвуч\w*|айтыңыз|айт\s|"
    r"оқып\s+беріңіз|оқып\s+бер|жіберіңіз|беріңіз)(?!\w)", re.IGNORECASE
)
_SENSITIVE_ITEM = re.compile(
    r"(?<!\w)(?:код\w*|кота|смс|sms|otp|одноразов\w*\s+парол\w*|"
    r"cvv|cvc|пин(?:-?код)?|pin|парол\w*\s+(?:от\s+)?(?:мобильн\w*\s+)?банк\w*|"
    r"логин\w*\s+и\s+парол\w*|данн\w*\s+карт\w*|номер\s+карт\w*|"
    r"бір\s+реттік\s+парол\w*|құпиясөз|құпия\s+сөз)(?!\w)", re.IGNORECASE
)
_NEGATED_DISCLOSURE = re.compile(
    r"(?<!\w)(?:не|никогда|нельзя|запрещено|ешқашан|айтпаңыз|айтпа|"
    r"бермеңіз|жібермеңіз)(?:\s+\w+){0,4}\s*$", re.IGNORECASE
)
_EXPLICIT_REQUEST_SIGNAL = "Просят продиктовать одноразовый код или банковские секреты"


def _has_explicit_secret_request(text: str) -> bool:
    """Catch direct code/password requests, including the ASR error «кота» for «код».

    The short ASR alias is only considered near a disclosure verb, so ordinary
    mentions of a cat do not trigger this signal.
    """
    normalized = " ".join(text.casefold().split())
    items = list(_SENSITIVE_ITEM.finditer(normalized))
    actions = list(_DISCLOSURE_ACTION.finditer(normalized))
    for item in items:
        for action in actions:
            if abs(item.start() - action.end()) > 56 and abs(action.start() - item.end()) > 56:
                continue
            before_action = normalized[max(0, action.start() - 36):action.start()]
            if _NEGATED_DISCLOSURE.search(before_action):
                continue
            if item.group(0) == "кота" and not re.search(r"продикт\w*|дикту\w*|прочит\w*", normalized):
                continue
            return True
    return False


def ensure_demo_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        demo = pd.DataFrame(
            [{"text": text, "label": label, "source": "synthetic_demo"} for text, label in DEMO_ROWS]
        )
        demo.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")


def load_dataset() -> pd.DataFrame:
    ensure_demo_dataset()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig").fillna("")
    for col in ["text", "label", "source"]:
        if col not in df.columns:
            df[col] = ""
    if AUGMENTATION_PATH.exists():
        extra = pd.read_csv(AUGMENTATION_PATH, encoding="utf-8-sig").fillna("")
        for col in ["text", "label", "source"]:
            if col not in extra.columns:
                extra[col] = "curated_synthetic" if col == "source" else ""
        df = pd.concat([df, extra[["text", "label", "source"]]], ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["source"] = df["source"].astype(str)
    return df[["text", "label", "source"]]


def save_dataset(df: pd.DataFrame) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = df[["text", "label", "source"]].copy()
    # Keep built-in synthetic augmentation in its separate, reviewable file.
    clean = clean[clean["source"] != "curated_synthetic"]
    clean.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    clean = df.copy()
    clean["text"] = clean["text"].astype(str).str.strip()
    clean["label"] = clean["label"].astype(str).str.lower().str.strip()
    clean = clean[clean["label"].isin(["scam", "legitimate"])]
    clean = clean[clean["text"].str.len() >= 10].drop_duplicates(subset=["text", "label"])
    counts = clean["label"].value_counts()
    if len(counts) < 2 or counts.min() < 4:
        raise ValueError("Нужно минимум по 4 разных примера каждого класса (scam и legitimate).")

    test_count = max(2 * len(counts), round(len(clean) * 0.25))
    test_size = min(0.4, test_count / len(clean))
    X_train, X_test, y_train, y_test = train_test_split(
        clean["text"], clean["label"], test_size=test_size, random_state=42, stratify=clean["label"]
    )

    model = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=50000, sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="scam", zero_division=0),
        "recall": recall_score(y_test, pred, pos_label="scam", zero_division=0),
        "f1": f1_score(y_test, pred, pos_label="scam", zero_division=0),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=["legitimate", "scam"]).tolist(),
        "class_counts": counts.to_dict(),
        "dataset_size": len(clean),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return metrics


def rule_signals(text: str) -> list[str]:
    lower = text.casefold()
    signals = [title for title, phrases in SIGNALS.items() if any(p.casefold() in lower for p in phrases)]
    if _has_explicit_secret_request(text):
        signals.append(_EXPLICIT_REQUEST_SIGNAL)
    return signals


def predict_text(text: str) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Модель не обучена. Перейди во вкладку «Обучение».")
    if not text.strip():
        raise ValueError("Пустой текст нельзя анализировать.")
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    scam_score = float(probabilities[classes.index("scam")])
    signals = rule_signals(text)
    # This is an experimental ranking, NOT a calibrated real-world fraud probability.
    score = round(scam_score * 70 + min(len(signals), 5) * 6)
    if _EXPLICIT_REQUEST_SIGNAL in signals:
        score = max(score, 78)
    score = max(0, min(100, score))
    if score >= 80:
        label = "Критический риск"
    elif score >= 60:
        label = "Высокий риск"
    elif score >= 35:
        label = "Подозрительный диалог"
    else:
        label = "Низкий риск"
    return {"score": score, "label": label, "signals": signals, "model_score": round(scam_score * 100, 1)}


@lru_cache(maxsize=3)
def _get_whisper_model(model_size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper не установлен. Выполни: pip install -r requirements-audio.txt") from exc
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes: bytes, filename: str, model_size: str = "base", language: str | None = None) -> dict[str, Any]:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    try:
        # CPU + int8 is a practical starting point for laptops without a supported CUDA setup.
        model = _get_whisper_model(model_size)
        segments, info = model.transcribe(temp_path, language=language, vad_filter=True, beam_size=3)
        chunks = []
        for seg in segments:
            timestamp = f"[{seg.start:0.1f}s] "
            chunks.append(timestamp + seg.text.strip())
        return {
            "text": "\n".join(chunks).strip(),
            "language": getattr(info, "language", "unknown"),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)
