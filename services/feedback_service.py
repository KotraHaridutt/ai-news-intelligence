import asyncio
import uuid

from services import cache_service, errors, observability
from services.cache_service import WatchStoreUnavailable

# --- A reader's 👍 / 👎 on an answer (feature 2, Section J) ---
# The page sends the answer_id it already has (never a trace id). The answer's record in Redis
# says which trace wrote it (#68), and the rating goes on that trace as feedback "user_rating":
# 1 for 👍, 0 for 👎, so its average on a dashboard is the share of 👍. A 👎 also puts the
# trace in the review queue (#71). Every failure has its own status and message (#69).

RATING_KEY = "user_rating"
MAX_COMMENT = 1000
SAVE_TIMEOUT_SECONDS = 15
SAVE_ATTEMPTS = 2
# One rating per answer: the feedback id comes from the answer id, and LangSmith overwrites a
# feedback created again with the same id (checked live in Section J), so a second click updates
# the first. uuid5 is a one-way hash: the answer id can't be read back from it.
_NAMESPACE = uuid.UUID("5f0c1d0e-7a51-4c55-9d1e-6a1f3b2c4d5e")


def feedback_id(answer_id: str) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"{answer_id}:{RATING_KEY}")


def _save(client, record: dict, answer_id: str, score: int, comment: str | None) -> bool:
    """Blocking (runs in a thread). Saves the rating; a 👎 also goes to the review queue. Returns whether it was queued."""
    client.create_feedback(trace_id=record["trace_id"], key=RATING_KEY, score=1 if score == 1 else 0, comment=comment,
                           feedback_id=feedback_id(answer_id), session_id=observability.project_id(record.get("project")),
                           source_info={"view": record.get("view")},
                           # Not upgraded to paid retention: reviewed examples are kept in the datasets (F) instead
                           extend_trace_retention=False, stop_after_attempt=SAVE_ATTEMPTS)
    if score != -1:
        return False
    # The rating is saved; the queue is a convenience, so its failure doesn't fail the request.
    # The run's full key (id, project, start time): a bare id is deprecated by the SDK after 2027-01-31.
    try:
        project = observability.project_id(record.get("project"))
        if not project or not record.get("trace_start"):
            raise ValueError("the answer's record has no project or start time")
        client.add_runs_to_annotation_queue(observability.review_queue_id(), runs=[
            {"run_id": record["trace_id"], "session_id": project, "start_time": record["trace_start"]}])
        return True
    except Exception as e:
        print(f"[WARN] 👎 saved, but its trace couldn't be added to the review queue: {type(e).__name__}: {e}")
        return False


async def rate(answer_id: str, score: int, comment: str | None) -> dict:
    """Raises AppError: 422 bad rating / comment too long, 404 expired, 409 not collected, 503 store / LangSmith."""
    if score not in (1, -1):
        raise errors.rating_invalid(f"score {score!r}")
    comment = (comment or "").strip() or None
    if comment and len(comment) > MAX_COMMENT:
        raise errors.comment_too_long(MAX_COMMENT, f"comment of {len(comment)} characters")
    if not cache_service.is_valid_id(answer_id):
        raise errors.rating_expired("malformed answer id")
    # Strict read: Redis down is "can't save right now", never a wrong "this answer expired"
    try:
        records = await cache_service.get_answers([answer_id], strict=True)
    except WatchStoreUnavailable as e:
        raise errors.rating_store_unavailable(str(e)) from e
    record = records.get(answer_id)
    if record is None:
        raise errors.rating_expired(f"answer {answer_id} expired or unknown")
    if not record.get("trace_id"):
        raise errors.rating_not_collected(f"answer {answer_id} ({record.get('view')}) has no trace")
    client = observability.client()
    if client is None:
        raise errors.ratings_off(f"tracing is off: {observability.status.get('reason')}")

    rated_ref = record["trace_id"].replace("-", "")[-8:]
    observability.annotate(rated_view=record.get("view"), rated_ref=rated_ref, rating=score, comment=bool(comment))
    try:
        queued = await asyncio.wait_for(asyncio.to_thread(_save, client, record, answer_id, score, comment), SAVE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as e:
        raise errors.rating_not_saved(f"LangSmith didn't answer in {SAVE_TIMEOUT_SECONDS}s") from e
    except Exception as e:
        raise errors.rating_not_saved(f"create_feedback failed: {type(e).__name__}: {e}") from e
    print(f"[INFO] RATING {'👍' if score == 1 else '👎'} on {record.get('view')} {rated_ref}"
          f"{' with a comment' if comment else ''}{', queued for review' if queued else ''}")
    observability.annotate(queued_for_review=queued)
    return {"saved": True, "score": score}
