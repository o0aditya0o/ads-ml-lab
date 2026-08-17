"""Database models. SQLite via SQLModel — small enough to read in one sitting."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Field, Session, SQLModel, create_engine

JUDGE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("JUDGE_DB", JUDGE / "data" / "judge.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False because uvicorn serves requests on a threadpool; SQLite's own
# locking still serialises writes.
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 15},
    echo=False,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record) -> None:
    """Settings SQLite does not apply on its own, per connection.

    None of these are defaults, and an earlier version of this file described WAL in a
    comment without ever enabling it — the database was running in ``journal_mode=delete``
    with foreign keys unenforced. Pragmas are per-connection in SQLite, so they belong on
    the connect event, not in a one-off call at startup.

    - ``foreign_keys=ON``: SQLite writes FK constraints into the schema but ignores them
      unless asked. ``Submission.user_id`` references ``user.id``; without this a
      submission could outlive the account that made it.
    - ``journal_mode=WAL``: readers stop blocking on the writer, which is what makes a
      leaderboard read safe while a submission is being scored. Persistent once set, but
      harmless to repeat.
    - ``busy_timeout``: wait rather than raise "database is locked" when a write overlaps.
    - ``synchronous=NORMAL``: the usual pairing with WAL. Durable against process crashes;
      a power loss can cost the last transactions, which for a leaderboard is a fine trade.
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=15000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    display_name: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=utcnow)
    is_admin: bool = False
    is_system: bool = False          # the account that owns the seeded baselines


class Submission(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    week: int = Field(index=True)
    label: str = ""                  # what the entrant called this model
    created_at: datetime = Field(default_factory=utcnow, index=True)

    status: str = "pending"          # pending | scored | rejected
    error: str = ""                  # why it was rejected, shown back to the entrant

    n_rows: int = 0
    public_score: float | None = Field(default=None, index=True)
    private_score: float | None = None
    metrics_json: str = "{}"         # full bundle on the public rows
    private_metrics_json: str = "{}"

    is_baseline: bool = False


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        _migrate_missing_columns(s)


def _migrate_missing_columns(session: Session) -> None:
    """Add columns introduced after a database already existed.

    A real project would use Alembic. For a single-file SQLite judge with two tables,
    this is honest about what it is: additive-only, and it never drops or rewrites.
    """
    from sqlalchemy import text

    for table, model in (("user", User), ("submission", Submission)):
        existing = {r[1] for r in session.exec(text(f"PRAGMA table_info({table})")).all()}
        if not existing:
            continue
        for name, field in model.model_fields.items():
            if name in existing:
                continue
            ann = field.annotation
            sql_type = ("INTEGER" if ann in (int, bool) or ann == (int | None)
                        else "FLOAT" if ann in (float,) or ann == (float | None)
                        else "TEXT")
            session.exec(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
            session.commit()


def get_session() -> Session:
    return Session(engine)
