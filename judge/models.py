"""Database models. SQLite via SQLModel — small enough to read in one sitting."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine

JUDGE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("JUDGE_DB", JUDGE / "data" / "judge.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False because uvicorn serves requests on a threadpool; SQLite's own
# locking still serialises writes. WAL keeps readers from blocking on the writer.
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=False,
)


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
