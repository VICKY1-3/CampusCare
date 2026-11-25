import os
import re
import math
from collections import Counter
import psycopg
from psycopg.rows import dict_row
from functools import wraps
from datetime import datetime, timedelta, timezone
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import click
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, abort, g
)
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-only-change-this-before-deployment"
)

DATABASE_URL = os.environ.get("DATABASE_URL")

DUPLICATE_SIMILARITY_THRESHOLD = 0.35

STATUSES = ("Submitted", "In Progress", "Resolved", "Rejected")
ACTIVE_STATUSES = ("Submitted", "In Progress")
PRIORITIES = ("High", "Medium", "Low")
CATEGORIES = (
    "Infrastructure", "Academics", "Hostel",
    "Library", "IT Support", "Other"
)
SLA_HOURS = {
    "High": 48,      # 48 hours for High / Emergency
    "Medium": 96,    # 4 days for Medium
    "Low": 168       # 7 days for Low
}

DEPARTMENT_SCORING_RULES = {
    "Examination / Admin Cell": {
        "strong": re.compile(
            r"\b(admit\s*card|hall\s*ticket|exam\s*(?:form|schedule|timetable|clash|centre|center|postponed|cancelled|result)|marks?|grade\s*sheet|transcript|migration\s*certificate|bonafide|degree\s*certificate|revaluation|re-?checking|backlog|attendance\s*shortage|enrollment|enrolment|admission\s*cell)\b",
            re.IGNORECASE,
        ),
        "moderate": re.compile(
            r"\b(exam|exams|examination|result|certificate|id\s*card|registration|semester|syllabus|faculty|professor|lecture|class\s*schedule)\b",
            re.IGNORECASE,
        ),
        "categories": {"Academics"},
    },
    "IT & Lab Infrastructure": {
        "strong": re.compile(
            r"\b(wi-?fi|internet|network|ethernet|router|portal|login\s*error|server\s*down|computer\s*lab|pc\s*not\s*working|projector|software\s*license|erp)\b",
            re.IGNORECASE,
        ),
        "moderate": re.compile(
            r"\b(computer|pc|laptop|lab|system|software|website|app|account|password|printer|scanner)\b",
            re.IGNORECASE,
        ),
        "categories": {"IT Support"},
    },
    "Estate / Maintenance Department": {
        "strong": re.compile(
            r"\b(no\s*water|water\s*(?:supply|leak|cut|shortage)|power\s*(?:cut|outage|failure)|electricity|switchboard|short\s*circuit|wiring|plumbing|washroom|toilet|sewage|drainage|fan|ac|air\s*conditioner|roof\s*leak|hostel\s*room|mess\s*food)\b",
            re.IGNORECASE,
        ),
        "moderate": re.compile(
            r"\b(hostel|water|light|lights|fan|room|mess|warden|furniture|bench|door|window|cleaning|sanitation|maintenance|repair)\b",
            re.IGNORECASE,
        ),
        "categories": {"Hostel", "Infrastructure"},
    },
    "Library & Academic Resources": {
        "strong": re.compile(
            r"\b(library\s*card|book\s*(?:unavailable|shortage|issue|return|fine)|reading\s*room|reference\s*book|journal|e-?journal|library\s*fine)\b",
            re.IGNORECASE,
        ),
        "moderate": re.compile(
            r"\b(library|book|books|journal|magazine|reading)\b",
            re.IGNORECASE,
        ),
        "categories": {"Library"},
    },
    "Accounts Department": {
        "strong": re.compile(
            r"\b(fee\s*(?:receipt|refund|payment|dues|structure)|tuition\s*fee|scholarship|challan|overcharged|double\s*payment|security\s*deposit)\b",
            re.IGNORECASE,
        ),
        "moderate": re.compile(
            r"\b(fee|fees|payment|refund|receipt|fine|dues|billing|amount)\b",
            re.IGNORECASE,
        ),
        "categories": set(),
    },
}

# Comprehensive High-Severity Keywords List (100+ terms + common phrase variants)
HIGH_SEVERITY_KEYWORDS = [
    # Ragging, Harassment & Safety
    "ragging", "harassment", "bullying", "mental torture", "threat", "assault",
    "violence", "stalking", "abuse", "discrimination", "safety", "hazard",
    "danger", "risk", "unsafe", "vulnerability", "security breach", "unauthorized entry",

    # Fire, Electrical & Utility Outages
    "fire", "spark", "sparking", "short circuit", "broken electrical", "switchboard",
    "exposed wire", "live wire", "power failure", "power outage", "electricity outage",
    "blackout", "transformer blast", "burning smell", "smoke", "malfunctioning socket",
    "fuse blown", "no water", "zero water", "water cut off", "water supply cut",
    "water outage", "water shortage", "no electricity", "wi-fi down", "wifi down",
    "internet down", "network down", "completely down", "internet outage",
    "wifi outage", "wi-fi outage", "system down", "server down", "connection lost",

    # Water, Sanitation & Biohazards
    "water contamination", "contaminated", "poison", "toxic", "sewage overflow",
    "drainage blockage", "stagnant water", "cholera", "disease outbreak",
    "unclean drinking water", "drinking water", "washroom", "toilet", "sanitation",
    "hygiene", "rotten food", "insect infestation", "pest menace",
    "biohazard", "hazardous waste", "chemical leak", "gas leak",

    # Structural Integrity & Collapse Risks
    "collapse", "structural damage", "cracked wall", "falling ceiling", "roof leak",
    "broken staircase", "loose balcony", "shattered glass", "blocked emergency exit",
    "faulty elevator", "stuck lift", "damaged foundation",

    # Medical, Academic & Physical Emergencies
    "injury", "injured", "accident", "bleeding", "unconscious", "medical emergency",
    "ambulance", "severe allergic reaction", "food poisoning", "electric shock",
    "burn", "fracture", "fainting", "suicide threat", "depression crisis",
    "exam clash", "exam cancelled", "exam postponed", "exams tomorrow",

    # Critical Campus Incidents & Crimes
    "theft", "robbery", "burglary", "stolen", "vandalism", "property damage",
    "weapon", "drugs", "alcohol consumption", "ragging complaint", "extortion",
    "ragging incident", "emergency", "urgent", "critical", "sos", "immediate action"
]

HIGH_SEVERITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(term).replace(r"\ ", r"[\s-]+")
        for term in HIGH_SEVERITY_KEYWORDS
    ) + r")\b",
    re.IGNORECASE
)

OUTAGE_DOWN_PATTERN = re.compile(
    r"\b(?:wi-?fi|wifi|internet|network|power|water\s+supply)\b.{0,25}\b(?:down|outage|lost|cut\s+off|unavailable)\b",
    re.IGNORECASE,
)

# Comprehensive Low-Severity / General Inquiry Keywords List (100+ terms)
LOW_SEVERITY_KEYWORDS = [
    # General Inquiries & Questions
    "inquiry", "enquiry", "question", "doubt", "clarification", "information",
    "details", "guidelines", "procedure", "process", "schedule", "timetable",
    "timing", "calendar", "notification", "circular", "announcement", "updates",

    # Suggestions & Feedback
    "suggestion", "feedback", "proposal", "recommendation", "idea", "opinion",
    "improvement", "enhancement", "upgrade request", "wishlist", "review",
    "appreciation", "compliment", "comment", "thought",

    # General Requests & Amenities
    "request", "general", "optional", "preference", "convenience", "access",
    "permission", "concession", "pass", "card reissue", "identity card replacement",
    "locker", "parking space", "bicycle rack", "drinking water station addition",
    "shade", "seating arrangement", "bench", "lighting adjustment",

    # Sports, Cultural & Extracurricular
    "sports equipment", "gym equipment", "cultural event", "fest", "club activity",
    "society", "competition", "tournament", "practice session", "auditorium booking",
    "seminar hall", "exhibition", "newsletter", "magazine", "membership",

    # Administrative Assistance & Minor Adjustments
    "form submission", "document collection", "certificate copy", "bonafide request",
    "fee structure doubt", "library card renewal", "book exchange", "reading room seating",
    "wifi password update", "email alias", "portal assistance", "profile correction",
    "address update", "contact change", "spelling correction", "minor adjustment",

    # Non-Urgent Feedback
    "routine check", "periodic maintenance", "cleaning schedule", "dustbin placement",
    "signboard", "notice board", "directional board", "campus landscaping",
    "plant watering", "lawn maintenance", "mosquito repellent check", "general upkeep"
]

LOW_SEVERITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(term).replace(r"\ ", r"[\s-]+")
        for term in LOW_SEVERITY_KEYWORDS
    ) + r")\b",
    re.IGNORECASE
)

# Labeled campus complaint corpus for lightweight TF-IDF + Multinomial Naive Bayes classifiers
LABELED_TRAINING_SAMPLES = [
    # (text, department, priority)
    ("Live exposed electrical wire sparking near hostel staircase", "Estate / Maintenance Department", "High"),
    ("No water supply in boys hostel washrooms since morning", "Estate / Maintenance Department", "High"),
    ("Short circuit and burning smell from classroom switchboard", "Estate / Maintenance Department", "High"),
    ("Sewage overflow outside ground floor washroom creating health hazard", "Estate / Maintenance Department", "High"),
    ("Roof leaking heavily onto electrical panels during rain", "Estate / Maintenance Department", "High"),
    ("Ceiling fan is not working in classroom 204", "Estate / Maintenance Department", "Medium"),
    ("Broken wooden bench and loose window latch in lecture hall", "Estate / Maintenance Department", "Medium"),
    ("Hostel room tube light flickering and needs replacement", "Estate / Maintenance Department", "Medium"),
    ("Air conditioner cooling low in seminar room", "Estate / Maintenance Department", "Low"),
    ("Request to add extra dustbins and benches near garden lawn", "Estate / Maintenance Department", "Low"),

    ("Campus Wi-Fi and student portal server completely down before submission", "IT & Lab Infrastructure", "High"),
    ("Unauthorized access or security breach on student login portal", "IT & Lab Infrastructure", "High"),
    ("Computer lab systems failing to boot and Ethernet disconnected", "IT & Lab Infrastructure", "Medium"),
    ("Classroom projector HDMI port malfunctioning during lectures", "IT & Lab Infrastructure", "Medium"),
    ("Software compiler license expired on lab desktops", "IT & Lab Infrastructure", "Medium"),
    ("Request Wi-Fi password reset and institutional email alias update", "IT & Lab Infrastructure", "Low"),
    ("Suggestion to install dark mode and faster search on student portal", "IT & Lab Infrastructure", "Low"),

    ("Two core semester exams scheduled at exact same time exam clash", "Examination / Admin Cell", "High"),
    ("Admit card not generated and hall ticket missing one day before exam", "Examination / Admin Cell", "High"),
    ("Ragging and harassment threat reported outside academic block", "Examination / Admin Cell", "High"),
    ("Internal marks discrepancy and grade sheet revaluation request", "Examination / Admin Cell", "Medium"),
    ("Attendance shortage calculation error in semester portal", "Examination / Admin Cell", "Medium"),
    ("Request for bonafide certificate and migration certificate copy", "Examination / Admin Cell", "Low"),
    ("Inquiry about semester exam timetable and circular details", "Examination / Admin Cell", "Low"),
    ("Spelling correction needed on student ID card profile", "Examination / Admin Cell", "Low"),

    ("Reference textbooks for database systems out of stock in library", "Library & Academic Resources", "Medium"),
    ("Library reading room AC and power sockets not working", "Library & Academic Resources", "Medium"),
    ("E-journal portal access denied from campus network", "Library & Academic Resources", "Medium"),
    ("Inquiry regarding library card renewal and book return schedule", "Library & Academic Resources", "Low"),
    ("Suggestion to extend library reading room hours during exams", "Library & Academic Resources", "Low"),

    ("Double tuition fee payment deducted online receipt not generated", "Accounts Department", "Medium"),
    ("Scholarship disbursement pending despite approved verification", "Accounts Department", "Medium"),
    ("Security deposit refund delayed after course completion", "Accounts Department", "Medium"),
    ("Clarification needed on semester fee structure and challan deadline", "Accounts Department", "Low"),
    ("Request duplicate fee receipt copy for reimbursement", "Accounts Department", "Low"),
]

_ML_MODELS = None


def _get_ml_classifiers():
    """Lazily train and cache TF-IDF + Multinomial Naive Bayes classifiers for department and priority."""
    global _ML_MODELS
    if _ML_MODELS is not None:
        return _ML_MODELS
    if not SKLEARN_AVAILABLE:
        return None

    texts = [sample[0] for sample in LABELED_TRAINING_SAMPLES]
    departments = [sample[1] for sample in LABELED_TRAINING_SAMPLES]
    priorities = [sample[2] for sample in LABELED_TRAINING_SAMPLES]

    dept_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    dept_pipeline.fit(texts, departments)

    prio_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    prio_pipeline.fit(texts, priorities)

    _ML_MODELS = (dept_pipeline, prio_pipeline)
    return _ML_MODELS


def predict_complaint_ml(title, description, category=""):
    """
    Predict (department, dept_confidence, priority, prio_confidence) using TF-IDF + MultinomialNB.
    Returns None if scikit-learn is unavailable.
    """
    models = _get_ml_classifiers()
    if models is None:
        return None

    dept_model, prio_model = models
    query_text = f"{category} {title} {description}".strip()
    if not query_text:
        return None

    dept_probs = dept_model.predict_proba([query_text])[0]
    dept_idx = int(dept_probs.argmax())
    pred_dept = str(dept_model.classes_[dept_idx])
    dept_conf = float(dept_probs[dept_idx])

    prio_probs = prio_model.predict_proba([query_text])[0]
    prio_idx = int(prio_probs.argmax())
    pred_prio = str(prio_model.classes_[prio_idx])
    prio_conf = float(prio_probs[prio_idx])

    return {
        "department": pred_dept,
        "dept_confidence": round(dept_conf * 100, 1),
        "priority": pred_prio,
        "prio_confidence": round(prio_conf * 100, 1),
    }


def route_to_department(title, description, category, ml_prediction=None):
    """Score candidate departments using category affinity, strong/moderate rules, and ML confidence."""
    title_text = title or ""
    desc_text = description or ""

    scores = {dept: 0.0 for dept in DEPARTMENT_SCORING_RULES}
    scores["General Administration"] = 1.0 if category == "Other" else 0.0

    for dept, rules in DEPARTMENT_SCORING_RULES.items():
        if category in rules["categories"]:
            scores[dept] += 3.0

        strong_title_hits = len(rules["strong"].findall(title_text))
        strong_desc_hits = len(rules["strong"].findall(desc_text))
        mod_title_hits = len(rules["moderate"].findall(title_text))
        mod_desc_hits = len(rules["moderate"].findall(desc_text))

        scores[dept] += (strong_title_hits * 5.0) + (strong_desc_hits * 3.0)
        scores[dept] += (mod_title_hits * 2.0) + (mod_desc_hits * 1.0)

    # Blend supervised Naive Bayes prediction when confident (>= 45%)
    if ml_prediction and ml_prediction["dept_confidence"] >= 45.0:
        ml_dept = ml_prediction["department"]
        if ml_dept in scores:
            scores[ml_dept] += 2.5 * (ml_prediction["dept_confidence"] / 100.0)

    best_dept = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best_dept] == 0:
        category_fallbacks = {
            "Infrastructure": "Estate / Maintenance Department",
            "Hostel": "Estate / Maintenance Department",
            "Academics": "Examination / Admin Cell",
            "Library": "Library & Academic Resources",
            "IT Support": "IT & Lab Infrastructure",
            "Other": "General Administration",
        }
        return category_fallbacks.get(category, "General Administration")

    return best_dept


def compute_sla_breach_risk(complaint, now_utc=None, has_active_escalation=False):
    """
    Compute an SLA breach risk score (0-100) and risk label for open complaints.
    Returns None for closed/resolved/rejected tickets.
    """
    if complaint.get("status") not in ("Submitted", "In Progress"):
        return None

    now = now_utc or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    priority = complaint.get("priority") or "Medium"
    sla_window_hours = SLA_HOURS.get(priority, 96)
    sla_due_at = complaint.get("sla_due_at")

    if sla_due_at is not None:
        due_dt = sla_due_at
        if due_dt.tzinfo is None:
            due_dt = due_dt.replace(tzinfo=timezone.utc)
        remaining_hours = (due_dt - now).total_seconds() / 3600.0
        elapsed_ratio = max(0.0, min(1.25, (sla_window_hours - remaining_hours) / sla_window_hours))
    else:
        created_raw = complaint.get("created_at")
        elapsed_ratio = 0.25
        if created_raw:
            try:
                created_dt = datetime.strptime(str(created_raw)[:16], "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
                elapsed_hours = max(0.0, (now - created_dt).total_seconds() / 3600.0)
                elapsed_ratio = max(0.0, min(1.25, elapsed_hours / sla_window_hours))
            except ValueError:
                pass

    elapsed_points = min(55.0, (elapsed_ratio / 1.0) * 55.0)
    priority_points = {"High": 20.0, "Medium": 10.0, "Low": 4.0}.get(priority, 10.0)
    extra_upvotes = max(0, int(complaint.get("upvote_count") or 1) - 1)
    upvote_points = min(24.0, extra_upvotes * 8.0)
    escalation_points = 15.0 if has_active_escalation else 0.0

    score = int(round(min(100.0, elapsed_points + priority_points + upvote_points + escalation_points)))

    if score >= 75:
        level = "Critical"
        badge_class = "bg-danger"
    elif score >= 50:
        level = "High"
        badge_class = "bg-warning text-dark"
    elif score >= 25:
        level = "Medium"
        badge_class = "bg-info text-dark"
    else:
        level = "Low"
        badge_class = "bg-secondary"

    return {
        "score": score,
        "level": level,
        "badge_class": badge_class,
        "elapsed_pct": int(round(min(100.0, elapsed_ratio * 100.0))),
        "has_escalation": has_active_escalation,
    }


def detect_complaint_surges(db, window_days=14):
    """
    Compare each category's active/unresolved last-24-hour volume (excluding 'Resolved' and
    'Rejected') against its prior 14-day rolling daily baseline to surface ongoing spikes.
    """
    now_local = datetime.now()
    cutoff_24h = now_local - timedelta(hours=24)
    cutoff_baseline = now_local - timedelta(days=window_days + 1)

    rows = db.execute(
        """
        SELECT category, title, description, status, created_at
        FROM complaints
        WHERE created_at >= %s
          AND (
              created_at < %s
              OR status NOT IN ('Resolved', 'Rejected')
          )
        ORDER BY complaint_id DESC
        """,
        (
            cutoff_baseline.strftime("%Y-%m-%d %H:%M"),
            cutoff_24h.strftime("%Y-%m-%d %H:%M"),
        )
    ).fetchall()

    recent_by_cat = {cat: [] for cat in CATEGORIES}
    daily_buckets = {
        cat: {d: 0 for d in range(window_days)}
        for cat in CATEGORIES
    }

    for row in rows:
        cat = row["category"]
        if cat not in recent_by_cat:
            continue
        try:
            created_dt = datetime.strptime(str(row["created_at"])[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        if created_dt >= cutoff_24h:
            if row.get("status") not in ("Resolved", "Rejected"):
                recent_by_cat[cat].append(row)
        elif created_dt >= cutoff_baseline:
            day_offset = int((cutoff_24h - created_dt).total_seconds() // 86400)
            if 0 <= day_offset < window_days:
                daily_buckets[cat][day_offset] += 1

    surges = []
    for cat in CATEGORIES:
        recent_items = recent_by_cat[cat]
        recent_count = len(recent_items)
        if recent_count < 3:
            continue

        history_counts = list(daily_buckets[cat].values())
        baseline_avg = sum(history_counts) / float(window_days)
        variance = sum((c - baseline_avg) ** 2 for c in history_counts) / float(window_days)
        std_dev = math.sqrt(variance)

        if baseline_avg > 0:
            ratio = recent_count / baseline_avg
            z_score = (recent_count - baseline_avg) / std_dev if std_dev > 0 else ratio
            is_surge = (ratio >= 2.5 and z_score >= 2.0)
        else:
            ratio = float(recent_count)
            z_score = float(recent_count)
            is_surge = (recent_count >= 4)

        if is_surge:
            keyword_counts = Counter()
            for item in recent_items:
                for kw in set(extract_keywords(f"{item['title']} {item['description']}")):
                    keyword_counts[kw] += 1
            top_topics = [kw for kw, _ in keyword_counts.most_common(3)]

            surges.append({
                "category": cat,
                "recent_24h": recent_count,
                "baseline_daily": round(baseline_avg, 2),
                "multiplier": round(ratio, 1),
                "z_score": round(z_score, 1),
                "severity": "Critical" if (recent_count >= 6 or ratio >= 4.0) else "Elevated",
                "top_topics": top_topics,
            })

    surges.sort(key=lambda s: (s["recent_24h"], s["multiplier"]), reverse=True)
    return surges


def classify_complaint(title, description, category, requested_priority="", corroborating_count=0):
    """
    Combine supervised TF-IDF + Multinomial Naive Bayes predictions with regex safety guardrails.
    Returns (routed_department, priority, sla_due_at, was_downgraded_from_high).
    """
    combined_text = f"{title} {description}"
    ml_pred = predict_complaint_ml(title, description, category)

    # 1. Hybrid Weighted + ML Department Routing
    routed_department = route_to_department(title, description, category, ml_prediction=ml_pred)

    # 2. Hybrid Severity Scoring: Regex safety rules + outage patterns + corroboration protect High urgency
    has_regex_high = bool(
        HIGH_SEVERITY_PATTERN.search(combined_text)
        or OUTAGE_DOWN_PATTERN.search(combined_text)
    )
    has_ml_high = bool(ml_pred and ml_pred["priority"] == "High" and ml_pred["prio_confidence"] >= 65.0)
    has_high_signal = has_regex_high or (corroborating_count >= 2) or has_ml_high
    was_downgraded = False

    if requested_priority == "High":
        if has_high_signal:
            priority = "High"
        else:
            priority = "Medium"
            was_downgraded = True
    elif requested_priority in ("Medium", "Low"):
        priority = requested_priority
    elif has_high_signal:
        priority = "High"
    elif LOW_SEVERITY_PATTERN.search(combined_text) or (
        ml_pred and ml_pred["priority"] == "Low" and ml_pred["prio_confidence"] >= 55.0
    ):
        priority = "Low"
    else:
        priority = "Medium"

    # 3. SLA Deadline Calculation
    sla_due_at = datetime.now().astimezone() + timedelta(hours=SLA_HOURS[priority])

    return routed_department, priority, sla_due_at, was_downgraded


def extract_keywords(text):
    """Extract meaningful lowercase keywords (>3 chars) for fallback duplicate detection."""
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "there", "about", "which", "when", "where", "please",
        "issue", "problem", "complaint", "college", "campus", "need", "help"
    }
    words = re.findall(r"[a-z0-9\-]{4,}", text.lower())
    return [w for w in words if w not in stopwords]


def find_duplicate_complaint(new_text, recent_complaints, threshold=DUPLICATE_SIMILARITY_THRESHOLD):
    """Find the closest matching recent complaint using TF-IDF + cosine similarity, with keyword fallback."""
    if not recent_complaints:
        return None

    candidate_texts = [
        f"{row['title']} {row['description']}".strip()
        for row in recent_complaints
    ]

    if SKLEARN_AVAILABLE and any(candidate_texts):
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=2000
            )
            tfidf_matrix = vectorizer.fit_transform([new_text] + candidate_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            best_idx = int(similarities.argmax())
            best_score = float(similarities[best_idx])

            if best_score >= threshold:
                return recent_complaints[best_idx]
            return None
        except ValueError:
            pass

    new_keywords = set(extract_keywords(new_text))
    for existing in recent_complaints:
        existing_keywords = set(
            extract_keywords(f"{existing['title']} {existing['description']}")
        )
        common = new_keywords.intersection(existing_keywords)
        if len(common) >= 2:
            return existing

    return None


def generate_ai_insights(title, description, category=""):
    """Generate sentiment summary plus supervised ML classifier predictions."""
    text = f"{title} {description}".lower()

    frustration_words = [
        "angry", "bad", "terrible", "worst", "unacceptable",
        "foul", "danger", "health", "hazard", "useless"
    ]
    score = sum(1 for word in frustration_words if word in text)

    if score >= 2 or "emergency" in text or "ragging" in text:
        sentiment = "High Frustration / Critical"
        ai_badge = "🚨 Critical Alert"
    elif score == 1:
        sentiment = "Moderate Concern"
        ai_badge = "⚠️ Moderate Attention"
    else:
        sentiment = "Standard Inquiry"
        ai_badge = "ℹ️ General"

    summary = description[:140] + "..." if len(description) > 140 else description
    ml_prediction = predict_complaint_ml(title, description, category)

    return {
        "sentiment": sentiment,
        "badge": ai_badge,
        "ai_summary": summary,
        "ml": ml_prediction,
    }


def get_db():
    if "db" not in g:
        if not DATABASE_URL:
            raise RuntimeError(
                "Set the DATABASE_URL environment variable first."
            )

        g.db = psycopg.connect(
            DATABASE_URL,
            row_factory=dict_row
        )

    return g.db


def log_security_event(event_type, student_id=None, details=None):
    """Record a security/audit event without storing passwords."""
    try:
        with psycopg.connect(
            DATABASE_URL,
            row_factory=dict_row
        ) as db:
            db.execute(
                """
                INSERT INTO security_audit_logs
                    (event_type, student_id, ip_address, details)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    event_type,
                    student_id,
                    request.remote_addr,
                    details
                )
            )
            db.commit()
    except psycopg.Error:
        app.logger.exception("Could not save security audit event.")


def get_recent_priority_downgrade_count(db, user_id):
    """Count how many times a student's requested High priority was downgraded in the last 30 days."""
    user_row = db.execute(
        "SELECT student_id FROM users WHERE user_id = %s",
        (user_id,)
    ).fetchone()
    if not user_row or not user_row["student_id"]:
        return 0

    cutoff = datetime.now().astimezone() - timedelta(days=30)
    row = db.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM security_audit_logs
        WHERE student_id = %s
          AND event_type = 'priority_downgraded'
          AND created_at >= %s
        """,
        (user_row["student_id"], cutoff)
    ).fetchone()
    return int(row["cnt"]) if row else 0


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    ) as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student'
                    CHECK (role IN ('student', 'admin', 'committee'))
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                complaint_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Submitted',
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            ALTER TABLE complaints
            ADD COLUMN IF NOT EXISTS priority TEXT NOT NULL DEFAULT 'Medium'
        """)

        db.execute("""
            ALTER TABLE complaints
            ADD COLUMN IF NOT EXISTS routed_department TEXT NOT NULL DEFAULT 'General Administration'
        """)

        db.execute("""
            ALTER TABLE complaints
            ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ
        """)

        db.execute("""
            ALTER TABLE complaints
            ADD COLUMN IF NOT EXISTS upvote_count INTEGER NOT NULL DEFAULT 1
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS complaint_upvotes (
                upvote_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                complaint_id INTEGER NOT NULL REFERENCES complaints(complaint_id),
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (complaint_id, user_id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS complaint_updates (
                update_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                complaint_id INTEGER NOT NULL
                    REFERENCES complaints(complaint_id),
                admin_id INTEGER NOT NULL REFERENCES users(user_id),
                status TEXT NOT NULL,
                remarks TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)

        # Approved college student roster
        db.execute("""
            CREATE TABLE IF NOT EXISTS approved_students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT NOT NULL,
                is_registered BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)

        db.execute("""
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS student_id TEXT
        """)

        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            users_student_id_unique
            ON users (student_id)
        """)

        approved_students = [
            ("colaamna", "Aamna"),
            ("colabhishek", "Abhishek Dewangan"),
            ("colamar", "Amar Gupta"),
            ("colharshita", "Harshita Singh"),
            ("colisha", "Isha Patel"),
            ("colmenka", "Menka Sachdev"),
            ("colnagesh", "Nagesh Baghel"),
            ("colnidhi", "Nidhi Verma"),
            ("colnisha", "Nisha Chauhan"),
            ("colpallavi", "Pallavi Sahu"),
            ("colpawan", "Pawan Kumar Shukla"),
            ("colpushkar", "Pushkar Yadav"),
            ("colshubhanshu", "Shubhanshu Nayak"),
            ("colsunidhi", "Sunidhi Mishra"),
            ("coltanisha", "Tanisha"),
            ("colvivek", "Vivek Yadav"),
        ]

        db.execute("""
            CREATE TABLE IF NOT EXISTS security_audit_logs (
                log_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                event_type TEXT NOT NULL,
                student_id TEXT,
                ip_address INET,
                details TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.cursor().executemany("""
            INSERT INTO approved_students
                (student_id, student_name)
            VALUES (%s, %s)
            ON CONFLICT (student_id)
            DO UPDATE SET student_name = EXCLUDED.student_name
        """, approved_students)

        db.execute("""
            CREATE TABLE IF NOT EXISTS complaint_escalations (
                escalation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                complaint_id INTEGER NOT NULL
                    REFERENCES complaints(complaint_id),
                requested_by INTEGER REFERENCES users(user_id),
                escalation_type TEXT NOT NULL
                 CHECK (escalation_type IN ('student_appeal', 'direct_referral')),
                reason TEXT NOT NULL,
                assigned_to INTEGER REFERENCES users(user_id),
                status TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (status IN (
                        'Pending',
                        'Under Review',
                        'Referred to Committee',
                        'Decision Issued',
                        'Closed'
                    )),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.execute("""
            ALTER TABLE complaint_escalations
            ADD COLUMN IF NOT EXISTS action_completed BOOLEAN NOT NULL DEFAULT TRUE
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS grievance_officers (
                officer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                designation TEXT NOT NULL
                    CHECK (designation IN ('Officer', 'Committee Member')),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, designation)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS escalation_history (
                history_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                escalation_id BIGINT NOT NULL
                    REFERENCES complaint_escalations(escalation_id),
                actor_id INTEGER REFERENCES users(user_id),
                action TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalations_complaint
            ON complaint_escalations(complaint_id)
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_escalations_status
            ON complaint_escalations(status)
        """)

        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_escalations_one_active_per_complaint
            ON complaint_escalations(complaint_id)
            WHERE status IN (
                'Pending',
                'Under Review',
                'Referred to Committee'
            )
        """)


@app.cli.command("init-db")
def init_db_command():
    """Create the database tables."""
    init_db()
    click.echo("Database initialized.")


@app.cli.command("create-admin")
def create_admin_command():
    """Create an administrator account securely."""
    name = click.prompt("Admin name")
    email = click.prompt("Admin email").strip().lower()
    password = click.prompt(
        "Admin password",
        hide_input=True,
        confirmation_prompt=True
    )

    if len(password) < 8:
        raise click.ClickException(
            "Password must be at least 8 characters."
        )

    init_db()

    try:
        with psycopg.connect(DATABASE_URL) as db:
            db.execute(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (%s, %s, %s, 'admin')
                """,
                (name, email, generate_password_hash(password))
            )

        click.echo("Admin account created.")

    except psycopg.errors.UniqueViolation:
        raise click.ClickException(
            "That email is already registered."
        )


@app.cli.command("create-grievance-account")
def create_grievance_account_command():
    """Create a Committee Member account."""
    name = click.prompt("Account holder name").strip()
    email = click.prompt("Account email").strip().lower()

    password = click.prompt(
        "Account password",
        hide_input=True,
        confirmation_prompt=True
    )

    if not name:
        raise click.ClickException("Name cannot be empty.")

    if len(password) < 8:
        raise click.ClickException(
            "Password must be at least 8 characters."
        )

    init_db()

    try:
        with psycopg.connect(DATABASE_URL) as db:
            row = db.execute(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (%s, %s, %s, 'committee')
                RETURNING user_id
                """,
                (
                    name,
                    email,
                    generate_password_hash(password)
                )
            ).fetchone()

            user_id = row[0]

            db.execute(
                """
                INSERT INTO grievance_officers
                    (user_id, designation, is_active)
                VALUES (%s, 'Committee Member', TRUE)
                """,
                (user_id,)
            )

        click.echo(
            "Committee Member account created successfully."
        )

    except psycopg.errors.UniqueViolation:
        raise click.ClickException(
            "That email is already registered."
        )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def committee_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))

        if session.get("role") != "committee":
            app.logger.warning(
                "Committee access denied: session role=%r",
                session.get("role")
            )
            abort(403)

        member = get_db().execute(
            """
            SELECT officer_id
            FROM grievance_officers
            WHERE user_id = %s
              AND designation = 'Committee Member'
              AND is_active = TRUE
            """,
            (session["user_id"],)
        ).fetchone()

        if member is None:
            app.logger.warning(
                "Committee access denied: no active committee record for user_id=%r",
                session.get("user_id")
            )
            abort(403)

        return view(*args, **kwargs)

    return wrapped


@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    if session["role"] == "committee":
        return redirect(url_for("committee_dashboard"))
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
@limiter.limit("50 per hour")
def register():
    if request.method == "POST":
        name = " ".join(
            request.form.get("name", "").strip().split()
        )
        student_id = request.form.get(
            "student_id", ""
        ).strip().lower()
        email = request.form.get(
            "email", ""
        ).strip().lower()
        password = request.form.get("password", "")

        if not name or not student_id or not email or not password:
            flash("All fields are required.")
            return render_template("register.html")

        if len(name) > 100 or len(student_id) > 30:
            flash("Name or student ID is too long.")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return render_template("register.html")

        db = get_db()

        try:
            student = db.execute(
                """
                SELECT student_id, student_name, is_registered
                FROM approved_students
                WHERE student_id = %s
                FOR UPDATE
                """,
                (student_id,)
            ).fetchone()

            if student is None:
                db.rollback()
                log_security_event(
                    "invalid_student_id",
                    student_id=student_id,
                    details="Registration attempted with an unapproved student ID"
                )
                flash("This student ID is not on the approved college list.")
                return render_template("register.html")

            if " ".join(student["student_name"].casefold().split()) != name.casefold():
                db.rollback()
                log_security_event(
                    "student_name_mismatch",
                    student_id=student_id,
                    details="Registration name did not match the approved student record"
                )
                flash("The name does not match the approved student record.")
                return render_template("register.html")

            if student["is_registered"]:
                db.rollback()
                log_security_event(
                    "duplicate_student_id",
                    student_id=student_id,
                    details="Registration attempted with an already registered student ID"
                )
                flash("This college student ID has already been registered.")
                return render_template("register.html")

            db.execute(
                """
                UPDATE approved_students
                SET is_registered = TRUE
                WHERE student_id = %s
                """,
                (student_id,)
            )

            db.execute(
                """
                INSERT INTO users
                    (name, email, password_hash, student_id)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    student["student_name"],
                    email,
                    generate_password_hash(password),
                    student_id
                )
            )

            db.commit()
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))

        except psycopg.errors.UniqueViolation:
            db.rollback()
            flash("That email or student ID is already registered.")

        except psycopg.Error:
            db.rollback()
            app.logger.exception("Student registration database error.")
            flash("Registration could not be completed. Please try again.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        selected_role = request.form.get("role", "student")

        user = get_db().execute(
            "SELECT * FROM users WHERE email = %s",
            (email,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            actual_role = user["role"]

            if selected_role == "admin" and actual_role != "admin":
                flash("Access denied: This account is not an Administrator.")
                return render_template("login.html")

            if selected_role == "committee" and actual_role != "committee":
                flash("Access denied: This account is not a Committee member.")
                return render_template("login.html")

            if selected_role == "student" and actual_role != "student":
                flash("Admins and Committee must use their respective login tabs.")
                return render_template("login.html")

            session.clear()
            session["user_id"] = user["user_id"]
            session["name"] = user["name"]
            session["role"] = actual_role

            if actual_role == "admin":
                return redirect(url_for("admin_dashboard"))
            elif actual_role == "committee":
                return redirect(url_for("committee_dashboard"))
            else:
                return redirect(url_for("dashboard"))

        log_security_event(
            "failed_login",
            details="Login attempted with invalid credentials"
        )
        flash("Invalid email or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if session["role"] != "student":
        abort(403)

    db = get_db()
    complaints = db.execute(
        """SELECT * FROM complaints WHERE user_id = %s
           ORDER BY complaint_id DESC""",
        (session["user_id"],)
    ).fetchall()

    return render_template(
        "dashboard.html",
        complaints=complaints,
        statuses=STATUSES
    )


@app.route("/complaint/new", methods=["GET", "POST"])
@login_required
@limiter.limit("30 per hour")
def new_complaint():
    if session["role"] != "student":
        abort(403)

    db = get_db()
    recent_downgrades = get_recent_priority_downgrade_count(db, session["user_id"])

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        category = request.form.get("category", "")
        requested_priority = request.form.get("priority", "").strip()
        bypass_duplicate = request.form.get("bypass_duplicate", "") == "true"

        if not title or not description or category not in CATEGORIES:
            flash("Enter all fields and choose a valid category.")
            return render_template(
                "new_complaint.html",
                categories=CATEGORIES,
                recent_downgrades=recent_downgrades
            )
        elif len(title) > 120 or len(description) > 5000:
            flash("Title or description is too long.")
            return render_template(
                "new_complaint.html",
                categories=CATEGORIES,
                recent_downgrades=recent_downgrades
            )

        recent_complaints = db.execute(
            """
            SELECT complaint_id, title, description, category, created_at, upvote_count
            FROM complaints
            WHERE category = %s
              AND user_id != %s
              AND status IN ('Submitted', 'In Progress')
              AND created_at >= %s
            ORDER BY complaint_id DESC
            """,
            (
                category,
                session["user_id"],
                (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
            )
        ).fetchall()

        matched_ticket = find_duplicate_complaint(
            f"{title} {description}",
            recent_complaints
        )

        if not bypass_duplicate and matched_ticket:
            return render_template(
                "new_complaint.html",
                categories=CATEGORIES,
                duplicate_match=matched_ticket,
                recent_downgrades=recent_downgrades,
                form_data={
                    "title": title,
                    "description": description,
                    "category": category,
                    "priority": requested_priority
                }
            )

        corroborating_count = max(
            len(recent_complaints),
            int(matched_ticket["upvote_count"]) if matched_ticket else 0
        )

        routed_dept, priority, sla_due_at, was_downgraded = classify_complaint(
            title, description, category, requested_priority, corroborating_count
        )

        cur = db.execute(
            """INSERT INTO complaints
               (user_id, title, description, category,
                status, created_at, priority, routed_department, sla_due_at, upvote_count)
               VALUES (%s, %s, %s, %s, 'Submitted', %s, %s, %s, %s, 1)
               RETURNING complaint_id""",
            (
                session["user_id"], title, description,
                category, datetime.now().strftime("%Y-%m-%d %H:%M"),
                priority, routed_dept, sla_due_at
            )
        )
        complaint_id = cur.fetchone()["complaint_id"]

        db.execute(
            """
            INSERT INTO complaint_upvotes (complaint_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT (complaint_id, user_id) DO NOTHING
            """,
            (complaint_id, session["user_id"])
        )

        if was_downgraded:
            user_row = db.execute(
                "SELECT student_id FROM users WHERE user_id = %s",
                (session["user_id"],)
            ).fetchone()
            student_sid = user_row["student_id"] if user_row else None
            db.execute(
                """
                INSERT INTO security_audit_logs
                    (event_type, student_id, ip_address, details)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    "priority_downgraded",
                    student_sid,
                    request.remote_addr,
                    f"CMP-{complaint_id:05d}: requested High -> set to Medium (no severe keywords or multi-student corroboration)"
                )
            )

        db.commit()

        if was_downgraded:
            flash(
                f"Complaint submitted (Reference: CMP-{complaint_id:05d}). Routed to: {routed_dept}. "
                "Marked as Medium priority (automatic high-priority requires safety/critical keywords or multiple reports; Admin can adjust if needed)."
            )
        else:
            flash(f"Complaint submitted successfully. Routed to: {routed_dept} [{priority} Priority]. Reference: CMP-{complaint_id:05d}")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    return render_template(
        "new_complaint.html",
        categories=CATEGORIES,
        recent_downgrades=recent_downgrades
    )


@app.route("/complaint/<int:complaint_id>/upvote", methods=["POST"])
@login_required
def upvote_complaint(complaint_id):
    if session["role"] != "student":
        abort(403)

    db = get_db()
    complaint = db.execute(
        "SELECT complaint_id, user_id FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    if complaint["user_id"] == session["user_id"]:
        flash("You cannot upvote your own complaint.")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    try:
        db.execute(
            """
            INSERT INTO complaint_upvotes (complaint_id, user_id)
            VALUES (%s, %s)
            """,
            (complaint_id, session["user_id"])
        )
        db.execute(
            """
            UPDATE complaints
            SET upvote_count = upvote_count + 1
            WHERE complaint_id = %s
            """,
            (complaint_id,)
        )
        db.commit()
        flash("Successfully upvoted/subscribed to this complaint!")
    except psycopg.errors.UniqueViolation:
        db.rollback()
        flash("You have already upvoted this complaint.")
    except psycopg.Error:
        db.rollback()
        app.logger.exception("Failed to upvote complaint %s", complaint_id)
        flash("Could not process upvote. Please try again.")

    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/complaint/<int:complaint_id>/escalate", methods=["POST"])
@login_required
@limiter.limit("5 per hour")
def request_escalation(complaint_id):
    if session.get("role") != "student":
        abort(403)

    reason = request.form.get("reason", "").strip()
    if not reason or len(reason) > 1000:
        flash("Please provide an escalation reason (maximum 1000 characters).")
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

    db = get_db()
    complaint = db.execute(
        "SELECT complaint_id, user_id, status FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    if complaint["user_id"] != session["user_id"]:
        abort(403)

    try:
        existing = db.execute(
            """
            SELECT escalation_id
            FROM complaint_escalations
            WHERE complaint_id = %s
              AND status IN (
                  'Pending',
                  'Under Review',
                  'Referred to Committee'
              )
            """,
            (complaint_id,)
        ).fetchone()

        if existing:
            flash("An active escalation already exists for this complaint.")
            return redirect(
                url_for("complaint_detail", complaint_id=complaint_id)
            )

        escalation = db.execute(
            """
            INSERT INTO complaint_escalations
                (
                    complaint_id,
                    requested_by,
                    escalation_type,
                    reason,
                    status
                )
            VALUES (%s, %s, 'student_appeal', %s, 'Pending')
            RETURNING escalation_id
            """,
            (complaint_id, session["user_id"], reason)
        ).fetchone()

        db.execute(
            """
            INSERT INTO escalation_history
                (escalation_id, actor_id, action, notes)
            VALUES (%s, %s, 'Escalation Requested', %s)
            """,
            (escalation["escalation_id"], session["user_id"], reason)
        )

        db.commit()
        flash("Escalation request submitted for Admin review.")

    except psycopg.Error:
        db.rollback()
        app.logger.exception(
            "Failed to submit student escalation for complaint %s",
            complaint_id
        )
        flash("Could not submit your escalation request. Please try again.")

    return redirect(
        url_for("complaint_detail", complaint_id=complaint_id)
    )


@app.route("/complaint/<int:complaint_id>")
@login_required
def complaint_detail(complaint_id):
    db = get_db()
    complaint = db.execute(
        "SELECT * FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    if session["role"] == "student":
        if complaint["user_id"] != session["user_id"] and complaint["status"] not in ("Submitted", "In Progress"):
            abort(403)

    updates = db.execute(
        """SELECT complaint_updates.*, users.name AS admin_name
           FROM complaint_updates
           JOIN users ON users.user_id = complaint_updates.admin_id
           WHERE complaint_updates.complaint_id = %s
           ORDER BY complaint_updates.update_id ASC""",
        (complaint_id,)
    ).fetchall()

    committee_members = db.execute("""
        SELECT u.user_id, u.name, u.email
        FROM grievance_officers AS go
        JOIN users AS u ON u.user_id = go.user_id
        WHERE go.designation = 'Committee Member'
          AND go.is_active = TRUE
          AND u.role = 'committee'
        ORDER BY u.name
    """).fetchall()

    active_escalation = db.execute(
        """
        SELECT
            ce.*,
            requester.name AS requester_name,
            assignee.name AS assignee_name
        FROM complaint_escalations AS ce
        LEFT JOIN users AS requester ON requester.user_id = ce.requested_by
        LEFT JOIN users AS assignee ON assignee.user_id = ce.assigned_to
        WHERE ce.complaint_id = %s
          AND (
              ce.status IN (
                  'Pending',
                  'Under Review',
                  'Referred to Committee'
              )
              OR (ce.status = 'Decision Issued' AND ce.action_completed = FALSE)
          )
        """,
        (complaint_id,)
    ).fetchone()

    escalation_history = db.execute(
        """
        SELECT
            eh.*,
            u.name AS actor_name
        FROM escalation_history AS eh
        JOIN users AS u ON u.user_id = eh.actor_id
        WHERE eh.escalation_id IN (
            SELECT ce.escalation_id
            FROM complaint_escalations AS ce
            WHERE ce.complaint_id = %s
        )
        ORDER BY eh.created_at ASC
        """,
        (complaint_id,)
    ).fetchall()

    ai_insights = generate_ai_insights(
        complaint["title"],
        complaint["description"],
        complaint["category"]
    )

    return render_template(
        "complaint_detail.html",
        complaint=complaint,
        updates=updates,
        committee_members=committee_members,
        active_escalation=active_escalation,
        escalation_history=escalation_history,
        now_utc=datetime.now().astimezone(),
        ai_insights=ai_insights
    )


@app.route("/admin/analytics")
@admin_required
def admin_analytics():
    db = get_db()

    status_rows = db.execute(
        """SELECT status, COUNT(*) AS total
           FROM complaints
           GROUP BY status"""
    ).fetchall()

    status_labels = [row["status"] for row in status_rows]
    status_values = [row["total"] for row in status_rows]

    category_rows = db.execute(
        """SELECT category, COUNT(*) AS total
           FROM complaints
           GROUP BY category
           ORDER BY total DESC, category ASC"""
    ).fetchall()

    category_labels = [row["category"] for row in category_rows]
    category_values = [row["total"] for row in category_rows]

    from datetime import date
    from calendar import month_abbr

    today = date.today()
    months = []

    for offset in range(5, -1, -1):
        month_index = today.month - offset
        year = today.year

        while month_index <= 0:
            month_index += 12
            year -= 1

        months.append((year, month_index))

    monthly_labels = [
        f"{month_abbr[month]} {year}" for year, month in months
    ]
    monthly_values = []

    for year, month in months:
        month_key = f"{year:04d}-{month:02d}"
        total = db.execute(
            """SELECT COUNT(*) FROM complaints
               WHERE substring(created_at FROM 1 FOR 7) = %s""",
            (month_key,)
        ).fetchone()["count"]
        monthly_values.append(total)

    total_complaints = db.execute(
        "SELECT COUNT(*) FROM complaints"
    ).fetchone()["count"]

    resolved_complaints = db.execute(
        "SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'"
    ).fetchone()["count"]

    pending_complaints = db.execute(
        """SELECT COUNT(*) FROM complaints
           WHERE status IN ('Submitted', 'In Progress')"""
    ).fetchone()["count"]

    resolution_rate = (
        round(resolved_complaints * 100 / total_complaints)
        if total_complaints else 0
    )

    surges = detect_complaint_surges(db)

    return render_template(
        "admin_analytics.html",
        total_complaints=total_complaints,
        resolved_complaints=resolved_complaints,
        pending_complaints=pending_complaints,
        resolution_rate=resolution_rate,
        status_labels=status_labels,
        status_values=status_values,
        category_labels=category_labels,
        category_values=category_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        surges=surges,
    )


@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()
    now_utc = datetime.now().astimezone()

    raw_complaints = db.execute(
        """
        SELECT complaints.*,
               users.name AS student_name,
               users.email AS student_email
        FROM complaints
        JOIN users
            ON users.user_id = complaints.user_id
        ORDER BY complaint_id DESC
        """
    ).fetchall()

    active_escalated_ids = {
        row["complaint_id"]
        for row in db.execute(
            """
            SELECT DISTINCT complaint_id
            FROM complaint_escalations
            WHERE status IN ('Pending', 'Under Review', 'Referred to Committee')
            """
        ).fetchall()
    }

    complaints = []
    for row in raw_complaints:
        item = dict(row)
        item["sla_risk"] = compute_sla_breach_risk(
            item,
            now_utc=now_utc,
            has_active_escalation=(item["complaint_id"] in active_escalated_ids)
        )
        complaints.append(item)

    counts = {}
    for status in STATUSES:
        counts[status] = db.execute(
            "SELECT COUNT(*) FROM complaints WHERE status = %s",
            (status,)
        ).fetchone()["count"]

    surges = detect_complaint_surges(db)

    return render_template(
        "admin_dashboard.html",
        complaints=complaints,
        statuses=STATUSES,
        counts=counts,
        now_utc=now_utc,
        surges=surges,
    )


@app.route("/committee/dashboard")
@committee_required
def committee_dashboard():
    db = get_db()

    escalations = db.execute(
        """
        SELECT
            ce.*,
            c.title AS complaint_title,
            c.description AS complaint_description,
            c.category AS complaint_category,
            c.status AS complaint_status,
            u.name AS student_name,
            u.email AS student_email
        FROM complaint_escalations AS ce
        JOIN complaints AS c
            ON c.complaint_id = ce.complaint_id
        JOIN users AS u
            ON u.user_id = c.user_id
        WHERE ce.assigned_to = %s
        ORDER BY ce.created_at DESC
        """,
        (session["user_id"],)
    ).fetchall()

    return render_template(
        "committee_dashboard.html",
        escalations=escalations
    )


@app.route("/admin/complaint/<int:complaint_id>/update", methods=["POST"])
@admin_required
def update_complaint(complaint_id):
    status = request.form.get("status", "")
    remarks = request.form.get("remarks", "").strip()
    new_priority = request.form.get("priority", "").strip()

    if status not in STATUSES or len(remarks) > 1000:
        flash("Invalid status or remarks.")
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    complaint = db.execute(
        "SELECT complaint_id, priority, user_id FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    old_priority = complaint["priority"] or "Medium"
    final_priority = new_priority if new_priority in PRIORITIES else old_priority
    priority_changed = final_priority != old_priority

    logged_remarks = remarks
    if priority_changed:
        priority_note = f"[Priority changed: {old_priority} → {final_priority}]"
        logged_remarks = f"{priority_note} {remarks}".strip()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    db.execute(
        "UPDATE complaints SET status = %s, priority = %s WHERE complaint_id = %s",
        (status, final_priority, complaint_id)
    )
    db.execute(
        """INSERT INTO complaint_updates
           (complaint_id, admin_id, status, remarks, updated_at)
           VALUES (%s, %s, %s, %s, %s)""",
        (complaint_id, session["user_id"], status, logged_remarks, now)
    )

    if priority_changed and old_priority == "High" and final_priority in ("Medium", "Low"):
        student_row = db.execute(
            "SELECT student_id FROM users WHERE user_id = %s",
            (complaint["user_id"],)
        ).fetchone()
        log_security_event(
            "priority_downgraded",
            student_id=student_row["student_id"] if student_row else None,
            details=f"CMP-{complaint_id:05d}: Admin changed priority {old_priority} -> {final_priority}"
        )

    db.commit()
    flash(f"Complaint CMP-{complaint_id:05d} updated.")
    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/admin/complaint/<int:complaint_id>/escalate", methods=["POST"])
@admin_required
def escalate_complaint(complaint_id):
    db = get_db()

    reason = request.form.get("reason", "").strip()
    assigned_to_raw = request.form.get("assigned_to", "").strip()

    if not reason or len(reason) > 1000:
        flash("Please provide an escalation reason (maximum 1000 characters).")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    if not assigned_to_raw.isdigit():
        flash("Please select a valid committee member.")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    assigned_to = int(assigned_to_raw)

    complaint = db.execute(
        "SELECT complaint_id FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    committee_member = db.execute(
        """
        SELECT go.user_id
        FROM grievance_officers AS go
        JOIN users AS u ON u.user_id = go.user_id
        WHERE go.user_id = %s
          AND go.designation = 'Committee Member'
          AND go.is_active = TRUE
          AND u.role = 'committee'
        """,
        (assigned_to,)
    ).fetchone()

    if committee_member is None:
        flash("The selected committee member is not active.")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    try:
        existing = db.execute(
            """
            SELECT escalation_id
            FROM complaint_escalations
            WHERE complaint_id = %s
              AND status IN (
                  'Pending',
                  'Under Review',
                  'Referred to Committee'
              )
            """,
            (complaint_id,)
        ).fetchone()

        if existing:
            flash("This complaint already has an active escalation.")
            return redirect(url_for("complaint_detail", complaint_id=complaint_id))

        escalation = db.execute(
            """
            INSERT INTO complaint_escalations
                (
                    complaint_id,
                    requested_by,
                    escalation_type,
                    reason,
                    assigned_to,
                    status
                )
            VALUES (%s, %s, 'direct_referral', %s, %s, 'Referred to Committee')
            RETURNING escalation_id
            """,
            (
                complaint_id,
                session["user_id"],
                reason,
                assigned_to,
            )
        ).fetchone()

        escalation_id = escalation["escalation_id"]

        db.execute(
            """
            INSERT INTO escalation_history
                (escalation_id, actor_id, action, notes)
            VALUES (%s, %s, %s, %s)
            """,
            (
                escalation_id,
                session["user_id"],
                "Directly Referred to Committee",
                reason
            )
        )

        db.commit()
        flash("Complaint successfully referred to the committee.")

    except psycopg.Error:
        db.rollback()
        app.logger.exception("Failed to escalate complaint %s", complaint_id)
        flash("Could not escalate the complaint. Please try again.")

    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/admin/escalation/<int:escalation_id>/review", methods=["POST"])
@admin_required
def review_escalation(escalation_id):
    db = get_db()

    escalation = db.execute(
        """
        SELECT escalation_id, complaint_id, status
        FROM complaint_escalations
        WHERE escalation_id = %s
        """,
        (escalation_id,)
    ).fetchone()

    if escalation is None:
        abort(404)

    complaint_id = escalation["complaint_id"]

    if escalation["status"] != "Pending":
        flash("Only pending escalation requests can be reviewed.")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    decision = request.form.get("decision", "").strip()
    notes = request.form.get("notes", "").strip()

    if len(notes) > 1000:
        flash("Remarks must be 1000 characters or fewer.")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    try:
        if decision == "accept":
            assigned_to_raw = request.form.get("assigned_to", "").strip()
            if not assigned_to_raw.isdigit():
                flash("Please select a valid committee member.")
                return redirect(url_for("complaint_detail", complaint_id=complaint_id))

            assigned_to = int(assigned_to_raw)
            member = db.execute(
                """
                SELECT go.user_id
                FROM grievance_officers AS go
                JOIN users AS u ON u.user_id = go.user_id
                WHERE go.user_id = %s
                  AND go.designation = 'Committee Member'
                  AND go.is_active = TRUE
                  AND u.role = 'committee'
                """,
                (assigned_to,)
            ).fetchone()

            if member is None:
                flash("The selected committee member is not active.")
                return redirect(url_for("complaint_detail", complaint_id=complaint_id))

            db.execute(
                """
                UPDATE complaint_escalations
                SET status = 'Referred to Committee',
                    assigned_to = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (assigned_to, escalation_id)
            )

            db.execute(
                """
                INSERT INTO escalation_history
                    (escalation_id, actor_id, action, notes)
                VALUES (%s, %s, 'Escalation Request Accepted', %s)
                """,
                (
                    escalation_id,
                    session["user_id"],
                    notes or "Escalation request approved and referred to Committee."
                )
            )
            flash("Escalation request accepted and referred to the Committee.")

        elif decision == "reject":
            if not notes:
                flash("Please provide a reason for rejecting the escalation request.")
                return redirect(url_for("complaint_detail", complaint_id=complaint_id))

            db.execute(
                """
                UPDATE complaint_escalations
                SET status = 'Closed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (escalation_id,)
            )

            db.execute(
                """
                INSERT INTO escalation_history
                    (escalation_id, actor_id, action, notes)
                VALUES (%s, %s, 'Escalation Request Rejected', %s)
                """,
                (escalation_id, session["user_id"], notes)
            )
            flash("Escalation request rejected and recorded.")

        else:
            flash("Invalid review decision.")
            return redirect(url_for("complaint_detail", complaint_id=complaint_id))

        db.commit()

    except psycopg.Error:
        db.rollback()
        app.logger.exception("Failed to review escalation %s", escalation_id)
        flash("Could not process the escalation review. Please try again.")

    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


@app.route("/committee/escalation/<int:escalation_id>/update", methods=["POST"])
@committee_required
def committee_update_escalation(escalation_id):
    db = get_db()

    action = request.form.get("action", "").strip()
    actor_id = session["user_id"]

    escalation = db.execute(
        """
        SELECT escalation_id, complaint_id, assigned_to, status, action_completed
        FROM complaint_escalations
        WHERE escalation_id = %s
          AND assigned_to = %s
        """,
        (escalation_id, actor_id)
    ).fetchone()

    if escalation is None:
        abort(404)

    current_status = escalation["status"]
    complaint_id = escalation["complaint_id"]

    try:
        if action == "review":
            if current_status not in ("Pending", "Referred to Committee"):
                flash("This escalation cannot be moved to review.")
                return redirect(url_for("committee_dashboard"))

            new_status = "Under Review"
            notes = "Committee started reviewing the escalation."

            db.execute(
                """
                UPDATE complaint_escalations
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (new_status, escalation_id)
            )

        elif action == "decision":
            if current_status != "Under Review":
                flash("Only escalations under review can receive a decision.")
                return redirect(url_for("committee_dashboard"))

            decision = request.form.get("decision", "").strip()
            remarks = request.form.get("remarks", "").strip()

            allowed_decisions = {"Action Required", "Rejected", "Resolved"}

            if decision not in allowed_decisions:
                flash("Please select a valid committee decision.")
                return redirect(url_for("committee_dashboard"))

            if not remarks or len(remarks) > 1000:
                flash("Decision remarks are required (maximum 1000 characters).")
                return redirect(url_for("committee_dashboard"))

            new_status = "Decision Issued"
            action_completed = decision != "Action Required"
            notes = f"Decision: {decision}. Remarks: {remarks}"

            db.execute(
                """
                UPDATE complaint_escalations
                SET status = %s,
                    action_completed = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (new_status, action_completed, escalation_id)
            )

            db.execute(
                """
                UPDATE complaints
                SET status = %s
                WHERE complaint_id = %s
                """,
                (
                    "Resolved" if decision == "Resolved"
                    else "Rejected" if decision == "Rejected"
                    else "In Progress",
                    complaint_id
                )
            )

        elif action == "complete_action":
            if current_status != "Decision Issued" or escalation["action_completed"]:
                flash("No pending corrective action to complete for this escalation.")
                return redirect(url_for("committee_dashboard"))

            completion_notes = request.form.get("completion_notes", "").strip()
            if not completion_notes or len(completion_notes) > 1000:
                flash("Please record completion details (maximum 1000 characters).")
                return redirect(url_for("committee_dashboard"))

            new_status = "Decision Issued"
            notes = f"Corrective Action Completed: {completion_notes}"

            db.execute(
                """
                UPDATE complaint_escalations
                SET action_completed = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (escalation_id,)
            )

        elif action == "close":
            if current_status != "Decision Issued":
                flash("Issue a decision before closing the escalation.")
                return redirect(url_for("committee_dashboard"))

            if not escalation["action_completed"]:
                flash("Record completion of the required corrective action before closing.")
                return redirect(url_for("committee_dashboard"))

            new_status = "Closed"
            notes = "Committee closed the escalation after verifying completion."

            db.execute(
                """
                UPDATE complaint_escalations
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE escalation_id = %s
                """,
                (new_status, escalation_id)
            )

        else:
            flash("Invalid committee action.")
            return redirect(url_for("committee_dashboard"))

        db.execute(
            """
            INSERT INTO escalation_history
                (escalation_id, actor_id, action, notes)
            VALUES (%s, %s, %s, %s)
            """,
            (escalation_id, actor_id, new_status, notes)
        )

        db.commit()
        flash(f"Escalation updated: {new_status}.")

    except psycopg.Error:
        db.rollback()
        app.logger.exception(
            "Failed to update escalation %s", escalation_id
        )
        flash("Could not update the escalation. Please try again.")

    return redirect(url_for("committee_dashboard"))


@app.errorhandler(429)
def ratelimit_handler(error):
    return render_template("429.html"), 429


@app.errorhandler(403)
def forbidden(error):
    return "Access denied: you do not have permission to view this page.", 403


def create_default_admin():
    """Automatically create a default admin account if none exists."""
    try:
        with psycopg.connect(DATABASE_URL) as db:
            existing = db.execute(
                "SELECT user_id FROM users WHERE role = 'admin' LIMIT 1"
            ).fetchone()

            if not existing:
                db.execute(
                    """
                    INSERT INTO users (name, email, password_hash, role)
                    VALUES (%s, %s, %s, 'admin')
                    """,
                    ("Admin User", "admin@campuscare.com", generate_password_hash("admin12345"))
                )
                db.commit()
                app.logger.info("Default admin account created: admin@campuscare.com / admin12345")
    except Exception:
        app.logger.exception("Could not auto-create default admin.")


def create_default_committee():
    """Automatically create a default committee account if none exists."""
    try:
        with psycopg.connect(DATABASE_URL) as db:
            existing = db.execute(
                "SELECT user_id FROM users WHERE email = %s LIMIT 1",
                ("committee@gmail.com",)
            ).fetchone()

            if not existing:
                row = db.execute(
                    """
                    INSERT INTO users (name, email, password_hash, role)
                    VALUES (%s, %s, %s, 'committee')
                    RETURNING user_id
                    """,
                    ("Committee", "committee@gmail.com", generate_password_hash("committee123"))
                ).fetchone()

                user_id = row[0]
                db.execute(
                    """
                    INSERT INTO grievance_officers (user_id, designation, is_active)
                    VALUES (%s, 'Committee Member', TRUE)
                    ON CONFLICT DO NOTHING
                    """,
                    (user_id,)
                )
                db.commit()
                app.logger.info("Default committee account created: committee@gmail.com / committee123")
    except Exception:
        app.logger.exception("Could not auto-create default committee.")


if __name__ == "__main__":
    init_db()
    create_default_admin()
    create_default_committee()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    try:
        init_db()
        create_default_admin()
        create_default_committee()
    except Exception as e:
        app.logger.error(f"Startup init error: {e}")
```[cite: 6]

---

### 2. `templates/admin_dashboard.html`

```html
{% extends "base.html" %}

{% block title %}Admin Dashboard | CampusCare{% endblock %}

{% block content %}

<!-- Admin header -->
<div class="dashboard-header d-flex flex-wrap justify-content-between align-items-center gap-3 mb-4">
    <div>
        <p class="section-eyebrow mb-1">ADMINISTRATION PORTAL</p>
        <h1 class="dashboard-title mb-2">Admin Dashboard</h1>
        <p class="text-muted mb-0">
            Monitor, review, manage college complaints, and track SLA deadlines.
        </p>
    </div>

    <div class="d-flex flex-wrap align-items-center gap-3">
        <a href="{{ url_for('admin_analytics') }}" class="btn btn-primary fw-semibold">
            📊 View Analytics &amp; Reports →
        </a>

        <div class="admin-welcome">
            <span class="admin-avatar">A</span>
            <div>
                <div class="fw-bold">{{ session.get('name') }}</div>
                <small class="text-muted">Administrator</small>
            </div>
        </div>
    </div>
</div>

<!-- Outage / Volume Surge Alert Banner (Active / Unresolved Only) -->
{% if surges %}
<div class="alert alert-danger border-danger shadow-sm mb-4" role="alert">
    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2">
        <strong>🚨 Potential Outage / Active Complaint Surge Detected</strong>
        <span class="badge bg-danger">{{ surges|length }} active unresolved spike{{ 's' if surges|length > 1 else '' }} (last 24h vs 14-day avg)</span>
    </div>
    <ul class="mb-0 ps-3">
        {% for s in surges %}
        <li class="mb-1">
            <strong>{{ s.category }}:</strong> {{ s.recent_24h }} active unresolved complaint{{ 's' if s.recent_24h > 1 else '' }} in the last 24h
            (14-day avg: {{ s.baseline_daily }}/day · <strong>{{ s.multiplier }}×</strong> normal)
            {% if s.top_topics %}
                — Active trending terms: <span class="font-monospace">{{ s.top_topics|join(', ') }}</span>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
</div>
{% endif %}

<!-- Statistics cards -->
<div class="row g-3 mb-4">

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-blue">▤</div>
            <p class="summary-label">Total Complaints</p>
            <h2 class="summary-number">{{ counts.values()|sum }}</h2>
            <p class="summary-note">All complaints received</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-amber">◷</div>
            <p class="summary-label">Submitted</p>
            <h2 class="summary-number">{{ counts.get('Submitted', 0) }}</h2>
            <p class="summary-note">Awaiting action</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-purple">↻</div>
            <p class="summary-label">In Progress</p>
            <h2 class="summary-number">{{ counts.get('In Progress', 0) }}</h2>
            <p class="summary-note">Currently being handled</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-green">✓</div>
            <p class="summary-label">Resolved</p>
            <h2 class="summary-number">{{ counts.get('Resolved', 0) }}</h2>
            <p class="summary-note">Successfully resolved</p>
        </div>
    </div>

</div>

<!-- Complaints table and filters -->
<div class="content-card">

    <div class="content-card-header">
        <div>
            <h2 class="content-title">All Complaints &amp; SLA Tracker</h2>
            <p class="text-muted mb-0">
                Review student submissions, department routing, priorities, and deadlines.
            </p>
        </div>

        <span class="total-pill" id="adminComplaintCount">
            {{ complaints|length }} records
        </span>
    </div>

    <!-- Search and filters -->
    <div class="complaint-filters">

        <div class="filter-search">
            <label for="adminComplaintSearch" class="visually-hidden">
                Search complaints
            </label>
            <input
                type="search"
                id="adminComplaintSearch"
                class="form-control"
                placeholder="Search title, reference, student, or department..."
                autocomplete="off">
        </div>

        <div class="filter-select">
            <label for="adminStatusFilter" class="visually-hidden">
                Filter by status
            </label>
            <select id="adminStatusFilter" class="form-select">
                <option value="">All statuses</option>
                <option value="Submitted">Submitted</option>
                <option value="In Progress">In Progress</option>
                <option value="Resolved">Resolved</option>
                <option value="Rejected">Rejected</option>
            </select>
        </div>

        <div class="filter-select">
            <label for="adminCategoryFilter" class="visually-hidden">
                Filter by category
            </label>
            <select id="adminCategoryFilter" class="form-select">
                <option value="">All categories</option>
                {% for category in complaints|map(attribute='category')|unique|sort %}
                <option value="{{ category }}">{{ category }}</option>
                {% endfor %}
            </select>
        </div>

        <button type="button" id="adminResetFilters"
                class="btn btn-outline-secondary">
            Reset
        </button>

    </div>

    <div class="filter-chips" role="group" aria-label="Quick status filters">
        <button type="button" class="filter-chip active" data-admin-chip-status="">All</button>
        <button type="button" class="filter-chip" data-admin-chip-status="Submitted">Submitted</button>
        <button type="button" class="filter-chip" data-admin-chip-status="In Progress">In Progress</button>
        <button type="button" class="filter-chip" data-admin-chip-status="Resolved">Resolved</button>
    </div>

    <div class="filter-results-text" id="adminFilterResultsText"
         aria-live="polite">
        Showing {{ complaints|length }} complaints
    </div>

    <div class="table-responsive">
        <table class="table complaint-table mb-0">
            <thead>
                <tr>
                    <th>Reference</th>
                    <th>Student</th>
                    <th>Complaint &amp; Department</th>
                    <th>Priority</th>
                    <th>SLA / Due &amp; Risk</th>
                    <th>Status</th>
                    <th>Action</th>
                </tr>
            </thead>

            <tbody>
            {% for c in complaints %}
                <tr class="admin-complaint-row"
                    data-title="{{ c.title|lower }}"
                    data-reference="CMP-{{ '%05d'|format(c.complaint_id) }}"
                    data-student="{{ c.student_name|lower }}"
                    data-email="{{ c.student_email|lower }}"
                    data-department="{{ c.routed_department|lower }}"
                    data-status="{{ c.status }}"
                    data-category="{{ c.category }}">

                    <td>
                        <span class="reference-number">
                            CMP-{{ "%05d"|format(c.complaint_id) }}
                        </span>
                        {% if c.upvote_count > 1 %}
                        <div><span class="badge bg-body text-body border mt-1">👍 {{ c.upvote_count }} students</span></div>
                        {% endif %}
                    </td>

                    <td>
                        <div class="complaint-title">{{ c.student_name }}</div>
                        <small class="text-muted">{{ c.student_email }}</small>
                    </td>

                    <td>
                        <div class="complaint-title">{{ c.title }}</div>
                        <small class="text-primary fw-semibold">🏢 {{ c.routed_department or 'General Administration' }}</small>
                    </td>

                    <td>
                        {% if c.priority == 'High' %}
                            <span class="badge bg-danger">High / Emergency</span>
                        {% elif c.priority == 'Low' %}
                            <span class="badge bg-secondary">Low</span>
                        {% else %}
                            <span class="badge bg-info text-dark">Medium</span>
                        {% endif %}
                    </td>

                    <td>
                        {% if c.sla_due_at %}
                            <div class="small fw-semibold {% if c.status in ['Submitted', 'In Progress'] and c.sla_due_at < now_utc %}text-danger{% else %}text-muted{% endif %}">
                                ⏳ {{ c.sla_due_at.strftime('%d %b %Y, %H:%M') }}
                            </div>
                            {% if c.status in ['Submitted', 'In Progress'] and c.sla_due_at < now_utc %}
                                <span class="badge bg-danger bg-opacity-10 text-danger border border-danger mt-1 font-monospace">OVERDUE</span>
                            {% endif %}
                        {% else %}
                            <span class="text-muted small">N/A</span>
                        {% endif %}

                        {% if c.sla_risk %}
                            <div class="mt-1 d-flex flex-wrap gap-1">
                                <span class="badge {{ c.sla_risk.badge_class }}"
                                      title="SLA elapsed: {{ c.sla_risk.elapsed_pct }}%">
                                    Risk {{ c.sla_risk.score }}/100 · {{ c.sla_risk.level }}
                                </span>
                                {% if c.sla_risk.has_escalation %}
                                    <span class="badge bg-warning text-dark font-monospace">ESCALATED</span>
                                {% endif %}
                            </div>
                        {% endif %}
                    </td>

                    <td>
                        {% if c.status == 'Submitted' %}
                            <span class="status-badge status-submitted">
                                <span class="status-dot"></span> Submitted
                            </span>
                        {% elif c.status == 'In Progress' %}
                            <span class="status-badge status-progress">
                                <span class="status-dot"></span> In Progress
                            </span>
                        {% elif c.status == 'Resolved' %}
                            <span class="status-badge status-resolved">
                                <span class="status-dot"></span> Resolved
                            </span>
                        {% elif c.status == 'Rejected' %}
                            <span class="status-badge status-rejected">
                                <span class="status-dot"></span> Rejected
                            </span>
                        {% else %}
                            <span class="status-badge status-submitted">
                                {{ c.status }}
                            </span>
                        {% endif %}
                    </td>

                    <td>
                        <a class="btn btn-view"
                           href="{{ url_for('complaint_detail',
                                            complaint_id=c.complaint_id) }}">
                            Manage →
                        </a>
                    </td>
                </tr>

            {% else %}
                <tr>
                    <td colspan="7">
                        <div class="empty-state">
                            <div class="empty-state-icon">▤</div>
                            <h3>No complaints yet</h3>
                            <p>Student complaints will appear here when submitted.</p>
                        </div>
                    </td>
                </tr>
            {% endfor %}

                <tr id="adminNoMatches" hidden>
                    <td colspan="7">
                        <div class="empty-state">
                            <div class="empty-state-icon">⌕</div>
                            <h3>No matching complaints found</h3>
                            <p>Try changing your search or filters.</p>
                        </div>
                    </td>
                </tr>

            </tbody>
        </table>
    </div>

</div>

<script>
document.addEventListener("DOMContentLoaded", function () {
    const statusSelect = document.getElementById("adminStatusFilter");
    const chips = Array.from(document.querySelectorAll("[data-admin-chip-status]"));
    const resetBtn = document.getElementById("adminResetFilters");

    if (!statusSelect || !chips.length) return;

    function syncAdminChips() {
        chips.forEach((chip) => {
            chip.classList.toggle("active", chip.dataset.adminChipStatus === statusSelect.value);
        });
    }

    chips.forEach((chip) => {
        chip.addEventListener("click", function () {
            statusSelect.value = chip.dataset.adminChipStatus;
            statusSelect.dispatchEvent(new Event("change"));
            syncAdminChips();
        });
    });

    statusSelect.addEventListener("change", syncAdminChips);
    if (resetBtn) {
        resetBtn.addEventListener("click", () => setTimeout(syncAdminChips, 0));
    }
});
</script>

{% endblock %}
```[cite: 7]

---

### 3. `templates/admin_analytics.html`

```html
{% extends "base.html" %}

{% block title %}Admin Analytics | CampusCare{% endblock %}

{% block content %}

<div class="dashboard-header d-flex flex-wrap justify-content-between align-items-center gap-3 mb-4">
    <div>
        <p class="section-eyebrow mb-1">ADMINISTRATION PORTAL</p>
        <h1 class="dashboard-title mb-2">Analytics Dashboard</h1>
        <p class="text-muted mb-0">
            Insights into college complaints and their resolution.
        </p>
    </div>

    <a href="{{ url_for('admin_dashboard') }}" class="btn btn-outline-secondary">
        ← Back to Admin Dashboard
    </a>
</div>

{% if surges %}
<div class="alert alert-danger border-danger shadow-sm mb-4" role="alert">
    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2">
        <strong>🚨 Anomaly Detection: Active Unresolved Surge{{ 's' if surges|length > 1 else '' }}</strong>
        <span class="badge bg-danger">Active unresolved (24h) vs 14-day rolling baseline</span>
    </div>
    <ul class="mb-0 ps-3">
        {% for s in surges %}
        <li class="mb-1">
            <strong>{{ s.category }}</strong> ({{ s.severity }}): <strong>{{ s.recent_24h }}</strong> active unresolved complaint{{ 's' if s.recent_24h > 1 else '' }} in last 24h
            vs. <strong>{{ s.baseline_daily }}/day</strong> baseline ({{ s.multiplier }}× · z={{ s.z_score }})
            {% if s.top_topics %}
                — Active Keywords: <span class="font-monospace">{{ s.top_topics|join(', ') }}</span>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
</div>
{% endif %}

<!-- Summary cards -->
<div class="row g-3 mb-4">

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-blue">▤</div>
            <p class="summary-label">Total Complaints</p>
            <h2 class="summary-number">{{ total_complaints }}</h2>
            <p class="summary-note">All recorded complaints</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-green">✓</div>
            <p class="summary-label">Resolved</p>
            <h2 class="summary-number">{{ resolved_complaints }}</h2>
            <p class="summary-note">Successfully resolved</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-amber">◷</div>
            <p class="summary-label">Pending / In Progress</p>
            <h2 class="summary-number">{{ pending_complaints }}</h2>
            <p class="summary-note">Still requiring attention</p>
        </div>
    </div>

    <div class="col-6 col-xl-3">
        <div class="summary-card">
            <div class="summary-icon icon-purple">%</div>
            <p class="summary-label">Resolution Rate</p>
            <h2 class="summary-number">{{ resolution_rate }}%</h2>
            <p class="summary-note">Of all recorded complaints</p>
        </div>
    </div>

</div>

<!-- Status and category charts -->
<div class="row g-4 mb-4">

    <div class="col-12 col-lg-5">
        <div class="content-card h-100">
            <div class="content-card-header">
                <div>
                    <h2 class="content-title">Complaint Status</h2>
                    <p class="text-muted mb-0">Current status distribution</p>
                </div>
            </div>
            <div class="analytics-chart-container">
                <canvas id="statusChart"
                        role="img"
                        aria-label="Doughnut chart of complaint statuses"></canvas>
                <p id="statusEmpty" class="text-muted text-center" hidden>
                    No complaints available to chart yet.
                </p>
            </div>
        </div>
    </div>

    <div class="col-12 col-lg-7">
        <div class="content-card h-100">
            <div class="content-card-header">
                <div>
                    <h2 class="content-title">Complaints by Category</h2>
                    <p class="text-muted mb-0">Distribution across departments</p>
                </div>
            </div>
            <div class="analytics-chart-container">
                <canvas id="categoryChart"
                        role="img"
                        aria-label="Bar chart of complaints by category"></canvas>
                <p id="categoryEmpty" class="text-muted text-center" hidden>
                    No complaints available to chart yet.
                </p>
            </div>
        </div>
    </div>

</div>

<!-- Monthly trend chart -->
<div class="content-card mb-4">
    <div class="content-card-header">
        <div>
            <h2 class="content-title">Monthly Complaint Trends</h2>
            <p class="text-muted mb-0">
                Complaints submitted over the last six months
            </p>
        </div>
    </div>

    <div class="analytics-chart-container analytics-trend-container">
        <canvas id="monthlyChart"
                role="img"
                aria-label="Line chart of monthly complaint submissions"></canvas>
    </div>
</div>

<div
    id="analytics-data"
    data-status-labels='{{ status_labels|tojson|forceescape }}'
    data-status-values='{{ status_values|tojson|forceescape }}'
    data-category-labels='{{ category_labels|tojson|forceescape }}'
    data-category-values='{{ category_values|tojson|forceescape }}'
    data-monthly-labels='{{ monthly_labels|tojson|forceescape }}'
    data-monthly-values='{{ monthly_values|tojson|forceescape }}'
></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function () {
    const chartData = document.getElementById("analytics-data");

    const statusLabels = JSON.parse(chartData.dataset.statusLabels);
    const statusValues = JSON.parse(chartData.dataset.statusValues);
    const categoryLabels = JSON.parse(chartData.dataset.categoryLabels);
    const categoryValues = JSON.parse(chartData.dataset.categoryValues);
    const monthlyLabels = JSON.parse(chartData.dataset.monthlyLabels);
    const monthlyValues = JSON.parse(chartData.dataset.monthlyValues);

    const statusCanvas = document.getElementById("statusChart");
    const categoryCanvas = document.getElementById("categoryChart");
    const monthlyCanvas = document.getElementById("monthlyChart");

    if (typeof Chart === "undefined") {
        document.querySelectorAll(".analytics-chart-container").forEach(
            function (container) {
                const message = document.createElement("p");
                message.className = "text-muted text-center";
                message.textContent =
                    "Charts could not load. Check your internet connection and refresh.";
                container.appendChild(message);
            }
        );
        return;
    }

    const charts = [];

    function getChartPalette() {
        const isDark = document.documentElement.getAttribute("data-bs-theme") === "dark";
        return {
            textColor: isDark ? "#cbd5e1" : "#475569",
            gridColor: isDark ? "rgba(148, 163, 184, 0.16)" : "rgba(15, 23, 42, 0.08)",
            doughnutBorder: isDark ? "#1e293b" : "#ffffff"
        };
    }

    function applyChartTheme() {
        const palette = getChartPalette();
        charts.forEach(function (chart) {
            if (chart.options.plugins?.legend?.labels) {
                chart.options.plugins.legend.labels.color = palette.textColor;
            }
            if (chart.options.scales?.x) {
                chart.options.scales.x.ticks.color = palette.textColor;
                chart.options.scales.x.grid.color = palette.gridColor;
            }
            if (chart.options.scales?.y) {
                chart.options.scales.y.ticks.color = palette.textColor;
                chart.options.scales.y.grid.color = palette.gridColor;
            }
            if (chart.config.type === "doughnut" && chart.data.datasets[0]) {
                chart.data.datasets[0].borderColor = palette.doughnutBorder;
            }
            chart.update();
        });
    }

    const initialPalette = getChartPalette();

    if (statusValues.reduce((sum, value) => sum + value, 0) === 0) {
        document.getElementById("statusEmpty").hidden = false;
        statusCanvas.hidden = true;
    } else {
        charts.push(new Chart(statusCanvas, {
            type: "doughnut",
            data: {
                labels: statusLabels,
                datasets: [{
                    data: statusValues,
                    backgroundColor: [
                        "#fbbf24", "#6366f1", "#22c55e", "#ef4444"
                    ],
                    borderWidth: 2,
                    borderColor: initialPalette.doughnutBorder
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { color: initialPalette.textColor }
                    }
                }
            }
        }));
    }

    if (categoryValues.reduce((sum, value) => sum + value, 0) === 0) {
        document.getElementById("categoryEmpty").hidden = false;
        categoryCanvas.hidden = true;
    } else {
        charts.push(new Chart(categoryCanvas, {
            type: "bar",
            data: {
                labels: categoryLabels,
                datasets: [{
                    label: "Complaints",
                    data: categoryValues,
                    backgroundColor: "#5b8def",
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: initialPalette.textColor },
                        grid: { color: initialPalette.gridColor }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0, color: initialPalette.textColor },
                        grid: { color: initialPalette.gridColor }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        }));
    }

    charts.push(new Chart(monthlyCanvas, {
        type: "line",
        data: {
            labels: monthlyLabels,
            datasets: [{
                label: "Complaints submitted",
                data: monthlyValues,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.12)",
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: initialPalette.textColor },
                    grid: { color: initialPalette.gridColor }
                },
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0, color: initialPalette.textColor },
                    grid: { color: initialPalette.gridColor }
                }
            },
            plugins: {
                legend: {
                    labels: { color: initialPalette.textColor }
                }
            }
        }
    }));

    const themeToggle = document.getElementById("darkModeToggle");
    if (themeToggle) {
        themeToggle.addEventListener("click", function () {
            setTimeout(applyChartTheme, 0);
        });
    }
});
</script>

{% endblock %}
```[cite: 8]