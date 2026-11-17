
import os
import re
import psycopg
from psycopg.rows import dict_row
from functools import wraps
from datetime import datetime, timedelta
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import click
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, abort, g
)
from werkzeug.security import generate_password_hash, check_password_hash

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

STATUSES = ("Submitted", "In Progress", "Resolved", "Rejected")
CATEGORIES = (
    "Infrastructure", "Academics", "Hostel",
    "Library", "IT Support", "Other"
)
SLA_HOURS = {
    "High": 48,      # 48 hours for High / Emergency
    "Medium": 96,    # 4 days for Medium
    "Low": 168       # 7 days for Low
}

DEPARTMENT_RULES = [
    ("Accounts Department", re.compile(
        r"\b(fee|fees|transaction|payment|refund|scholarship|receipt|fine|dues)\b",
        re.IGNORECASE
    )),
    ("Estate / Maintenance Department", re.compile(
        r"\b(hostel|water|electricity|power|wiring|switchboard|plumbing|washroom|fan|ac|leak|roof|furniture|wifi|wi-fi)\b",
        re.IGNORECASE
    )),
    ("Examination / Admin Cell", re.compile(
        r"\b(exam|exams|marks|grade|result|id card|admit card|hall ticket|certificate|transcript|attendance|registration)\b",
        re.IGNORECASE
    )),
    ("Library & Academic Resources", re.compile(
        r"\b(library|book|books|journal|reading room|syllabus)\b",
        re.IGNORECASE
    )),
    ("IT & Lab Infrastructure", re.compile(
        r"\b(computer|pc|lab|software|network|server|internet|projector|portal)\b",
        re.IGNORECASE
    )),
]

HIGH_SEVERITY_PATTERN = re.compile(
    r"\b(ragging|harassment|safety|hazard|fire|spark|short circuit|broken electrical|switchboard|water contamination|contaminated|toxic|collapse|injury|emergency|urgent|theft)\b",
    re.IGNORECASE
)

LOW_SEVERITY_PATTERN = re.compile(
    r"\b(inquiry|enquiry|suggestion|feedback|request|general)\b",
    re.IGNORECASE
)


def classify_complaint(title, description, category, requested_priority=""):
    """Determine department routing, severity score, and SLA deadline."""
    combined_text = f"{title} {description}"

    # 1. Keyword Auto-Routing (with category fallback)
    routed_department = None
    for dept_name, pattern in DEPARTMENT_RULES:
        if pattern.search(combined_text):
            routed_department = dept_name
            break

    if routed_department is None:
        category_fallbacks = {
            "Infrastructure": "Estate / Maintenance Department",
            "Hostel": "Estate / Maintenance Department",
            "Academics": "Examination / Admin Cell",
            "Library": "Library & Academic Resources",
            "IT Support": "IT & Lab Infrastructure",
            "Other": "General Administration"
        }
        routed_department = category_fallbacks.get(
            category, "General Administration"
        )

    # 2. Regex-Based Severity Scoring (respects explicit High input or detects keywords)
    if HIGH_SEVERITY_PATTERN.search(combined_text) or requested_priority == "High":
        priority = "High"
    elif requested_priority in ("Medium", "Low"):
        priority = requested_priority
    elif LOW_SEVERITY_PATTERN.search(combined_text):
        priority = "Low"
    else:
        priority = "Medium"

    # 3. SLA Deadline Calculation
    sla_due_at = datetime.now().astimezone() + timedelta(hours=SLA_HOURS[priority])

    return routed_department, priority, sla_due_at


def extract_keywords(text):
    """Extract meaningful lowercase keywords (>3 chars) for duplicate detection."""
    stopwords = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "there", "about", "which", "when", "where", "please",
        "issue", "problem", "complaint", "college", "campus", "need", "help"
    }
    words = re.findall(r"[a-z0-9\-]{4,}", text.lower())
    return [w for w in words if w not in stopwords]
def generate_ai_insights(title, description):
    """Generate a lightweight heuristic AI summary and sentiment score."""
    text = f"{title} {description}".lower()
    
    frustration_words = [
        "angry", "bad", "terrible", "worst", "unacceptable",
        "foul", "danger", "health", "hazard", "useless"
    ]
    score = sum(1 for word in frustration_words if word in text)
    
    if score >= 2 or "emergency" in text or "ragging" in text:
        sentiment = "High Frustration / Critical"
        ai_badge = "🔴 Critical Alert"
    elif score == 1:
        sentiment = "Moderate Concern"
        ai_badge = "🟡 Moderate Attention"
    else:
        sentiment = "Standard Inquiry"
        ai_badge = "🟢 General"
        
    summary = description[:140] + "..." if len(description) > 140 else description
    
    return {
        "sentiment": sentiment,
        "badge": ai_badge,
        "ai_summary": summary
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
    """Record a security event without storing passwords."""
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

        # Add student_id to existing users without deleting accounts
        db.execute("""
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS student_id TEXT
        """)

        # Prevent multiple accounts from using the same student ID
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            users_student_id_unique
            ON users (student_id)
        """)

        # Insert the 16 approved students
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

        # Record security-related events
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


        # Complaint escalation records
        
        
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

        # Authorized grievance officers and committee members
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

        # History of escalation actions and decisions
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

        # Speed up escalation lookups
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
@limiter.limit("5 per hour")
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
            # Lock the roster entry so two requests cannot claim it together.
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


            # Claim the approved ID and create the account in one transaction.
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
        selected_role = request.form.get("role", "student") # Catches student, admin, or committee from the form tab

        user = get_db().execute(
            "SELECT * FROM users WHERE email = %s",
            (email,)
        ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            actual_role = user["role"]

            # Security check: Ensure the user is logging into their correct designated tab portal
            if selected_role == "admin" and actual_role != "admin":
                flash("Access denied: This account is not an Administrator.")
                return render_template("login.html")
            
            if selected_role == "committee" and actual_role != "committee":
                flash("Access denied: This account is not a Committee member.")
                return render_template("login.html")

            if selected_role == "student" and actual_role != "student":
                flash("Admins and Committee must use their respective login tabs.")
                return render_template("login.html")

            # Login successful, set session variables
            session.clear()
            session["user_id"] = user["user_id"]
            session["name"] = user["name"]
            session["role"] = actual_role

            print("DEBUG LOGIN:", user["email"], repr(actual_role))

            # Redirect based on user role dashboard
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
@limiter.limit("5 per hour")
def new_complaint():
    if session["role"] != "student":
        abort(403)

    db = get_db()

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        category = request.form.get("category", "")
        requested_priority = request.form.get("priority", "").strip()
        bypass_duplicate = request.form.get("bypass_duplicate", "") == "true"

        if not title or not description or category not in CATEGORIES:
            flash("Enter all fields and choose a valid category.")
            return render_template("new_complaint.html", categories=CATEGORIES)
        elif len(title) > 120 or len(description) > 5000:
            flash("Title or description is too long.")
            return render_template("new_complaint.html", categories=CATEGORIES)

        # 1. Check for similar duplicate complaints in the last 24 hours
        if not bypass_duplicate:
            recent_complaints = db.execute(
                """
                SELECT complaint_id, title, description, category, created_at
                FROM complaints
                WHERE category = %s
                  AND status IN ('Submitted', 'In Progress')
                  AND created_at >= %s
                ORDER BY complaint_id DESC
                """,
                (category, (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"))
            ).fetchall()

            new_keywords = set(extract_keywords(f"{title} {description}"))
            matched_ticket = None

            for existing in recent_complaints:
                existing_keywords = set(extract_keywords(f"{existing['title']} {existing['description']}"))
                common = new_keywords.intersection(existing_keywords)
                if len(common) >= 2:  # 2 or more shared significant keywords
                    matched_ticket = existing
                    break

            if matched_ticket:
                return render_template(
                    "new_complaint.html",
                    categories=CATEGORIES,
                    duplicate_match=matched_ticket,
                    form_data={
                        "title": title,
                        "description": description,
                        "category": category,
                        "priority": requested_priority
                    }
                )

        # 2. Run Auto-Routing, Severity Scoring & SLA Calculation
        routed_dept, priority, sla_due_at = classify_complaint(
            title, description, category, requested_priority
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

        # Record author's initial upvote
        db.execute(
            """
            INSERT INTO complaint_upvotes (complaint_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT (complaint_id, user_id) DO NOTHING
            """,
            (complaint_id, session["user_id"])
        )
        db.commit()

        flash(f"Complaint submitted successfully. Routed to: {routed_dept} [{priority} Priority]. Reference: CMP-{complaint_id:05d}")
        return redirect(url_for("complaint_detail", complaint_id=complaint_id))

    return render_template("new_complaint.html", categories=CATEGORIES)
@app.route("/complaint/<int:complaint_id>/upvote", methods=["POST"])
@login_required
def upvote_complaint(complaint_id):
    if session["role"] != "student":
        abort(403)

    db = get_db()
    complaint = db.execute(
        "SELECT complaint_id FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

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

    if (session["role"] == "student"
            and complaint["user_id"] != session["user_id"]):
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

    # Generate lightweight AI insights
    ai_insights = generate_ai_insights(complaint["title"], complaint["description"])

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

    # Complaint counts by status
    status_rows = db.execute(
        """SELECT status, COUNT(*) AS total
           FROM complaints
           GROUP BY status"""
    ).fetchall()

    status_labels = [row["status"] for row in status_rows]
    status_values = [row["total"] for row in status_rows]

    # Complaint counts by category
    category_rows = db.execute(
        """SELECT category, COUNT(*) AS total
           FROM complaints
           GROUP BY category
           ORDER BY total DESC, category ASC"""
    ).fetchall()

    category_labels = [row["category"] for row in category_rows]
    category_values = [row["total"] for row in category_rows]

    # Monthly complaint submissions for the last 6 months,
    # including months with zero complaints.
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

    # Summary figures
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
    )

@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()

    complaints = db.execute(
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

    counts = {}
    for status in STATUSES:
        counts[status] = db.execute(
            "SELECT COUNT(*) FROM complaints WHERE status = %s",
            (status,)
        ).fetchone()["count"]

    return render_template(
        "admin_dashboard.html",
        complaints=complaints,
        statuses=STATUSES,
        counts=counts,
        now_utc=datetime.now().astimezone()
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



@app.route("/admin/complaint/<int:complaint_id>/update",
           methods=["POST"])
@admin_required
def update_complaint(complaint_id):
    status = request.form.get("status", "")
    remarks = request.form.get("remarks", "").strip()

    if status not in STATUSES or len(remarks) > 1000:
        flash("Invalid status or remarks.")
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    complaint = db.execute(
        "SELECT complaint_id FROM complaints WHERE complaint_id = %s",
        (complaint_id,)
    ).fetchone()

    if complaint is None:
        abort(404)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    db.execute(
        "UPDATE complaints SET status = %s WHERE complaint_id = %s",
        (status, complaint_id)
    )
    db.execute(
        """INSERT INTO complaint_updates
           (complaint_id, admin_id, status, remarks, updated_at)
           VALUES (%s, %s, %s, %s, %s)""",
        (complaint_id, session["user_id"], status, remarks, now)
    )
    db.commit()
    flash(f"Complaint CMP-{complaint_id:05d} updated.")
    return redirect(url_for("complaint_detail",
                            complaint_id=complaint_id))


@app.route(
    "/admin/complaint/<int:complaint_id>/escalate",
    methods=["POST"]
)
@admin_required
def escalate_complaint(complaint_id):
    db = get_db()

    reason = request.form.get("reason", "").strip()
    assigned_to_raw = request.form.get("assigned_to", "").strip()

    if not reason or len(reason) > 1000:
        flash("Please provide an escalation reason (maximum 1000 characters).")
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

    if not assigned_to_raw.isdigit():
        flash("Please select a valid committee member.")
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

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
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

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
                    assigned_to,
                    status
                )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING escalation_id
            """,
            (
                complaint_id,
                session["user_id"],
                "direct_referral",
                reason,
                assigned_to,
                "Referred to Committee"
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

    return redirect(
        url_for("complaint_detail", complaint_id=complaint_id)
    )
@app.route(
    "/admin/escalation/<int:escalation_id>/review",
    methods=["POST"]
)
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
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

    decision = request.form.get("decision", "").strip()
    notes = request.form.get("notes", "").strip()

    if len(notes) > 1000:
        flash("Remarks must be 1000 characters or fewer.")
        return redirect(
            url_for("complaint_detail", complaint_id=complaint_id)
        )

    try:
        if decision == "accept":
            assigned_to_raw = request.form.get("assigned_to", "").strip()
            if not assigned_to_raw.isdigit():
                flash("Please select a valid committee member.")
                return redirect(
                    url_for("complaint_detail", complaint_id=complaint_id)
                )

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
                return redirect(
                    url_for("complaint_detail", complaint_id=complaint_id)
                )

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
                return redirect(
                    url_for("complaint_detail", complaint_id=complaint_id)
                )

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
            return redirect(
                url_for("complaint_detail", complaint_id=complaint_id)
            )

        db.commit()

    except psycopg.Error:
        db.rollback()
        app.logger.exception("Failed to review escalation %s", escalation_id)
        flash("Could not process the escalation review. Please try again.")

    return redirect(
        url_for("complaint_detail", complaint_id=complaint_id)
    )
@app.route(
    "/committee/escalation/<int:escalation_id>/update",
    methods=["POST"]
)
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
            if current_status not in (
                "Pending",
                "Referred to Committee"
            ):
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

            allowed_decisions = {
                "Action Required",
                "Rejected",
                "Resolved"
            }

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

if __name__ == "__main__":
    init_db()
    create_default_admin()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # This runs automatically when Gunicorn starts your app on Render!
    try:
        init_db()
        create_default_admin()
    except Exception as e:
        app.logger.error(f"Startup init error: {e}")
