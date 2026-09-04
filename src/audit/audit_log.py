"""
Payment Retry Sequencer — Phase 7: Audit Log
=============================================

Append-only, hash-chained audit log for every policy evaluation.

Design goals (see compliance_notes.md for the full rationale):

  1. APPEND-ONLY BY CONSTRUCTION. This class exposes exactly one write
     method: `record()`. There is no `update()`, no `delete()`, no
     `truncate()` — those verbs simply do not exist on this object, so
     no caller anywhere in the application can invoke them by accident
     or on purpose. Every physical write goes through a file descriptor
     opened with `O_APPEND`, which is a POSIX guarantee that the kernel
     always appends at end-of-file atomically, even if something in the
     process seeks the handle first.

  2. TAMPER-EVIDENT. Every record is chained to the previous one via a
     SHA-256 hash (`record_hash = sha256(prev_hash + this_record_json)`).
     `verify_chain()` recomputes the chain from scratch and will detect
     any edited, reordered, inserted, or deleted line — including edits
     made completely outside this class (e.g. someone hand-editing the
     file on disk).

  3. HONEST ABOUT ITS LIMITS. Hash-chaining plus O_APPEND is a strong,
     portable, pure-Python guarantee *within this process's write path*.
     It is tamper-EVIDENT, not tamper-PROOF: someone with filesystem
     write access and enough patience could still rewrite the whole
     file and regenerate a self-consistent (but fraudulent) chain. True
     WORM (write-once-read-many) guarantees require infrastructure
     controls outside of any Python process — e.g. `chattr +a` on
     Linux, a versioned + Object-Locked S3 bucket, or shipping records
     synchronously to a dedicated append-only log service. This module
     is the application-level half of a defense-in-depth story, not a
     replacement for that infra layer. See compliance_notes.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Iterator, Optional, Tuple

GENESIS_HASH = "0" * 64


class AuditLogWriteError(Exception):
    """Raised when a record could not be durably appended to disk."""


def _canonical(payload: dict) -> str:
    # sort_keys + default=str => stable, deterministic serialization so the
    # same logical record always hashes the same way.
    return json.dumps(payload, sort_keys=True, default=str)


class AuditLogger:
    """
    Append-only JSON-lines audit log, one evaluation per line.

    Public surface is intentionally minimal:
      - record(event: dict) -> dict         (the only write path)
      - iter_records() -> Iterator[dict]     (read-only)
      - verify_chain() -> (bool, Optional[int])  (read-only integrity check)

    No method on this class can modify or remove a line that has already
    been written.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        # Create the file up front (without truncating if it already
        # exists) so the very first append doesn't race on file creation.
        fd = os.open(self._path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        os.close(fd)

    # -- the only write path ------------------------------------------------

    def record(self, event: dict) -> dict:
        """
        Append one audit event and return the stored record (including
        its sequence number, prev_hash, and record_hash).

        `event` should be a JSON-serializable dict describing one policy
        evaluation. This method adds bookkeeping fields (`seq`,
        `prev_hash`, `record_hash`) — callers should not pass those keys
        themselves.
        """
        with self._lock:
            prev_hash = self._last_hash_locked()
            seq = self._count_lines_locked()

            payload = dict(event)
            payload["seq"] = seq
            payload["prev_hash"] = prev_hash
            body = _canonical(payload)
            payload["record_hash"] = hashlib.sha256(
                (prev_hash + body).encode("utf-8")
            ).hexdigest()

            line = _canonical(payload)
            try:
                fd = os.open(self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, (line + "\n").encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise AuditLogWriteError(
                    f"failed to append audit record to {self._path!r}: {exc}"
                ) from exc

            return payload

    # -- read-only helpers ----------------------------------------------------

    def iter_records(self) -> Iterator[dict]:
        """Read-only iteration over stored records, in append order."""
        if not os.path.exists(self._path):
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """
        Recompute the hash chain over the entire log file.

        Returns (True, None) if every record's hash and prev_hash link
        up correctly from the genesis hash onward. Returns (False, seq)
        naming the sequence number of the first record where the chain
        breaks (tampering, corruption, or a removed/reordered line).
        """
        expected_prev = GENESIS_HASH
        for record in self.iter_records():
            stored_hash = record.get("record_hash")
            unhashed = {k: v for k, v in record.items() if k != "record_hash"}
            recomputed = hashlib.sha256(
                (expected_prev + _canonical(unhashed)).encode("utf-8")
            ).hexdigest()
            if unhashed.get("prev_hash") != expected_prev or recomputed != stored_hash:
                return False, record.get("seq")
            expected_prev = stored_hash
        return True, None

    # -- private ---------------------------------------------------------------

    def _last_hash_locked(self) -> str:
        last = GENESIS_HASH
        for record in self.iter_records():
            h = record.get("record_hash")
            if h:
                last = h
        return last

    def _count_lines_locked(self) -> int:
        # O(n) per write. Fine for this exercise's volume; a production
        # deployment with high throughput should keep the running
        # sequence number and last hash in memory (or use a database
        # with an autoincrement key) instead of rescanning the file.
        count = 0
        if os.path.exists(self._path):
            with open(self._path, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1
        return count
