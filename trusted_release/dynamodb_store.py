"""Cross-run release lock and journal backed by a control-owned DynamoDB table.

The lock has no TTL. A runner crash leaves it in place until an operator checks
Buffer and the journal and removes that exact lock through a separate process.
There is deliberately no automatic stale-lock takeover.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from contextlib import contextmanager
from typing import Any, Iterator

from .executor import DurableReleaseStore, EPISODE, ReleaseHold, require


class DynamoDBReleaseStore(DurableReleaseStore):
    """A single-key DynamoDB table with strongly consistent reads.

    The caller supplies an authenticated boto3 DynamoDB *client*. Its IAM role
    must be limited to this control table. No credentials are read here.
    """

    def __init__(self, client: Any, table_name: str, namespace: str):
        require(client is not None, "DynamoDB client is unavailable")
        require(type(table_name) is str and re.fullmatch(r"[A-Za-z0-9_.-]{3,255}", table_name)
                is not None, "DynamoDB release table name is invalid")
        require(type(namespace) is str and re.fullmatch(r"[A-Za-z0-9_.-]{4,80}", namespace)
                is not None, "DynamoDB release namespace is invalid")
        self.client = client
        self.table_name = table_name
        self.namespace = namespace
        self._owner: str | None = None
        self._episode: str | None = None
        self._seen: dict[str, int | None] = {}

    @classmethod
    def from_aws(cls, table_name: str, namespace: str) -> "DynamoDBReleaseStore":
        """Use the job's AWS identity and region; never accept a producer token."""
        import boto3

        return cls(boto3.client("dynamodb"), table_name, namespace)

    def _key(self, episode_id: str, kind: str) -> str:
        require(type(episode_id) is str and EPISODE.fullmatch(episode_id) is not None,
                "release episode ID is invalid")
        return f"mool-katha-release/{self.namespace}/{episode_id}/{kind}"

    @contextmanager
    def exclusive(self, episode_id: str) -> Iterator[None]:
        require(self._owner is None, "release store is already holding a lock")
        key = self._key(episode_id, "lock")
        owner = secrets.token_hex(32)
        try:
            self.client.put_item(
                TableName=self.table_name,
                Item={"pk": {"S": key}, "owner": {"S": owner}},
                ConditionExpression="attribute_not_exists(pk)",
                ReturnConsumedCapacity="NONE",
            )
        except Exception as exc:
            raise ReleaseHold("cross-run release lock is held or acquisition is uncertain") from exc
        self._owner = owner
        self._episode = episode_id
        self._seen.clear()
        try:
            yield
        finally:
            self._owner = None
            self._episode = None
            self._seen.clear()
            try:
                self.client.delete_item(
                    TableName=self.table_name,
                    Key={"pk": {"S": key}},
                    ConditionExpression="#o = :owner",
                    ExpressionAttributeNames={"#o": "owner"},
                    ExpressionAttributeValues={":owner": {"S": owner}},
                )
            except Exception as exc:
                raise ReleaseHold("cross-run release lock removal is uncertain; inspect the lock") from exc

    def load(self, episode_id: str) -> dict[str, Any] | None:
        key = self._key(episode_id, "journal")
        try:
            result = self.client.get_item(
                TableName=self.table_name, Key={"pk": {"S": key}}, ConsistentRead=True)
            item = result.get("Item")
            if item is None:
                if self._episode == episode_id:
                    self._seen[episode_id] = None
                return None
            require(type(item) is dict and set(item) == {"pk", "revision", "sha256", "document"}
                    and item["pk"] == {"S": key}, "durable release journal item is malformed")
            revision = int(item["revision"]["N"])
            require(revision > 0 and str(revision) == item["revision"]["N"],
                    "durable release journal revision is malformed")
            raw = item["document"]["S"]
            require(type(raw) is str and 0 < len(raw.encode("utf-8")) <= 8192 and
                    item["sha256"] == {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
                    "durable release journal checksum differs")
            value = json.loads(raw)
            require(type(value) is dict, "durable release journal document is malformed")
            if self._episode == episode_id:
                self._seen[episode_id] = revision
            return value
        except ReleaseHold:
            raise
        except Exception as exc:
            raise ReleaseHold("durable release journal read is uncertain") from exc

    def save(self, episode_id: str, document: dict[str, Any]) -> None:
        require(self._owner is not None and self._episode == episode_id and
                episode_id in self._seen, "durable journal write requires the acquired release lock")
        try:
            raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ReleaseHold("durable release journal cannot be encoded") from exc
        require(0 < len(raw.encode("utf-8")) <= 8192, "durable release journal is oversized")
        previous = self._seen[episode_id]
        revision = 1 if previous is None else previous + 1
        journal_key = self._key(episode_id, "journal")
        lock_key = self._key(episode_id, "lock")
        condition = "attribute_not_exists(pk)" if previous is None else "#r = :previous"
        put: dict[str, Any] = {
            "TableName": self.table_name,
            "Item": {"pk": {"S": journal_key}, "revision": {"N": str(revision)},
                     "sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
                     "document": {"S": raw}},
            "ConditionExpression": condition,
        }
        if previous is not None:
            put["ExpressionAttributeNames"] = {"#r": "revision"}
            put["ExpressionAttributeValues"] = {":previous": {"N": str(previous)}}
        try:
            self.client.transact_write_items(TransactItems=[
                {"ConditionCheck": {
                    "TableName": self.table_name,
                    "Key": {"pk": {"S": lock_key}},
                    "ConditionExpression": "#o = :owner",
                    "ExpressionAttributeNames": {"#o": "owner"},
                    "ExpressionAttributeValues": {":owner": {"S": self._owner}},
                }},
                {"Put": put},
            ])
        except Exception as exc:
            raise ReleaseHold("durable release journal write is uncertain or conflicted") from exc
        self._seen[episode_id] = revision
