"""Simulate independent runners sharing one conditional DynamoDB table."""

from __future__ import annotations

import copy
import unittest

from trusted_release.dynamodb_store import DynamoDBReleaseStore
from trusted_release.executor import ReleaseHold


class MemoryDynamoClient:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.fail_transaction = False

    def put_item(self, *, Item, ConditionExpression, **_):
        key = Item["pk"]["S"]
        if key in self.items or ConditionExpression != "attribute_not_exists(pk)":
            raise ValueError("conditional lock conflict")
        self.items[key] = copy.deepcopy(Item)

    def get_item(self, *, Key, ConsistentRead, **_):
        assert ConsistentRead is True
        item = self.items.get(Key["pk"]["S"])
        return {"Item": copy.deepcopy(item)} if item is not None else {}

    def delete_item(self, *, Key, ExpressionAttributeValues, **_):
        key = Key["pk"]["S"]
        item = self.items.get(key)
        if item is None or item["owner"] != ExpressionAttributeValues[":owner"]:
            raise ValueError("conditional unlock conflict")
        del self.items[key]

    def transact_write_items(self, *, TransactItems):
        if self.fail_transaction:
            raise TimeoutError("uncertain write")
        lock = TransactItems[0]["ConditionCheck"]
        lock_item = self.items.get(lock["Key"]["pk"]["S"])
        if lock_item is None or lock_item["owner"] != lock["ExpressionAttributeValues"][":owner"]:
            raise ValueError("lock lost")
        put = TransactItems[1]["Put"]
        key = put["Item"]["pk"]["S"]
        old = self.items.get(key)
        if put["ConditionExpression"] == "attribute_not_exists(pk)":
            if old is not None:
                raise ValueError("journal already exists")
        elif old is None or old["revision"] != put["ExpressionAttributeValues"][":previous"]:
            raise ValueError("stale journal revision")
        self.items[key] = copy.deepcopy(put["Item"])


class DynamoDBStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = MemoryDynamoClient()
        self.first = DynamoDBReleaseStore(self.client, "control-release", "production-v1")
        self.second = DynamoDBReleaseStore(self.client, "control-release", "production-v1")

    def test_lock_is_cross_runner_and_journal_survives_runs(self):
        with self.first.exclusive("ep004"):
            self.assertIsNone(self.first.load("ep004"))
            with self.assertRaisesRegex(ReleaseHold, "lock is held"):
                with self.second.exclusive("ep004"):
                    pass
            self.first.save("ep004", {"create": {"state": "create_started"}})
            self.assertEqual(self.first.load("ep004"), {"create": {"state": "create_started"}})
        with self.second.exclusive("ep004"):
            self.assertEqual(self.second.load("ep004"), {"create": {"state": "create_started"}})
            self.second.save("ep004", {"create": {"state": "unknown_outcome"}})
        self.assertEqual(self.first.load("ep004"), {"create": {"state": "unknown_outcome"}})

    def test_save_requires_lock_and_conditional_owner(self):
        with self.assertRaisesRegex(ReleaseHold, "acquired release lock"):
            self.first.save("ep004", {})
        with self.assertRaisesRegex(ReleaseHold, "removal is uncertain"):
            with self.first.exclusive("ep004"):
                self.first.load("ep004")
                key = self.first._key("ep004", "lock")
                self.client.items[key]["owner"] = {"S": "operator-removed-owner"}
                with self.assertRaisesRegex(ReleaseHold, "uncertain or conflicted"):
                    self.first.save("ep004", {"create": None})

    def test_uncertain_write_cannot_be_treated_as_confirmed(self):
        with self.first.exclusive("ep004"):
            self.assertIsNone(self.first.load("ep004"))
            self.client.fail_transaction = True
            with self.assertRaisesRegex(ReleaseHold, "write is uncertain"):
                self.first.save("ep004", {"create": {"state": "create_started"}})
            self.assertIsNone(self.first.load("ep004"))

    def test_crashed_runner_lock_has_no_automatic_takeover(self):
        key = self.first._key("ep004", "lock")
        self.client.items[key] = {"pk": {"S": key}, "owner": {"S": "old-runner"}}
        with self.assertRaisesRegex(ReleaseHold, "lock is held"):
            with self.second.exclusive("ep004"):
                pass
        self.assertEqual(self.client.items[key]["owner"], {"S": "old-runner"})

    def test_tampered_journal_checksum_holds(self):
        with self.first.exclusive("ep004"):
            self.first.load("ep004")
            self.first.save("ep004", {"create": None})
        key = self.first._key("ep004", "journal")
        self.client.items[key]["document"] = {"S": '{"create":"tampered"}'}
        with self.assertRaisesRegex(ReleaseHold, "checksum differs"):
            self.second.load("ep004")


if __name__ == "__main__":
    unittest.main()
