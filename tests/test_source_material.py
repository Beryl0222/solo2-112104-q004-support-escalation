import json
import unittest
from pathlib import Path

DATA = Path(__file__).parents[1] / "data" / "support.json"
REQUIRED = {"id", "text"}
SYNTHETIC_FILES = [
    "support.json",
    "support_conversation.json",
    "support_ambiguous.json",
    "support_policy_update.json",
]


class SourceMaterialTests(unittest.TestCase):
    def test_synthetic_source_records_are_readable(self):
        for name in SYNTHETIC_FILES:
            with self.subTest(file=name):
                records = json.loads((DATA.parent / name).read_text(encoding="utf-8"))
                self.assertIsInstance(records, list)
                self.assertGreaterEqual(len(records), 4)
                self.assertTrue(all(REQUIRED <= record.keys() for record in records))
                self.assertTrue(
                    all(
                        isinstance(record["text"], str) and record["text"].strip()
                        for record in records
                    )
                )

    def test_turn_numbers_are_unique_and_ordered(self):
        for name in SYNTHETIC_FILES:
            with self.subTest(file=name):
                records = json.loads((DATA.parent / name).read_text(encoding="utf-8"))
                turns = [record["turn"] for record in records]
                self.assertEqual(turns, sorted(turns))
                self.assertEqual(len(turns), len(set(turns)))

    def test_policy_and_order_fixtures_are_synthetic(self):
        policies = json.loads((DATA.parent / "policies.json").read_text(encoding="utf-8"))
        orders = json.loads((DATA.parent / "orders.json").read_text(encoding="utf-8"))
        self.assertTrue(all("policy_id" in item and "conditions" in item for item in policies))
        self.assertTrue(orders)
        for order_id, fields in orders.items():
            self.assertIsInstance(order_id, str)
            self.assertIsInstance(fields, dict)
            for value in fields.values():
                self.assertNotIn("@", str(value))


if __name__ == "__main__":
    unittest.main()
