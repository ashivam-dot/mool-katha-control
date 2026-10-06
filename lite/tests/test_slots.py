import unittest
from datetime import datetime, timezone

from lite import slots

IST = slots.IST


def post(when: str, status: str = "scheduled") -> dict:
    return {"id": when, "status": status, "dueAt": datetime.fromisoformat(when).isoformat()}


class FreeSlotTest(unittest.TestCase):
    def test_first_lite_slot_today(self):
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], []), datetime(2026, 10, 6, 17, 30, tzinfo=IST))

    def test_old_lane_post_today_leaves_one_lite_slot(self):
        # ep020 is booked at 19:00 IST (13:30 UTC) on 6 October.
        old = [post("2026-10-06T13:30:00+00:00")]
        now = datetime(2026, 10, 6, 0, 10, tzinfo=IST)
        first = slots.free_slot(now, old, old)
        self.assertEqual(first, datetime(2026, 10, 6, 17, 30, tzinfo=IST))
        booked = old + [post(first.isoformat())]
        self.assertEqual(slots.free_slot(now, booked, booked), datetime(2026, 10, 7, 17, 30, tzinfo=IST))

    def test_two_shorts_a_day_across_lanes(self):
        day = [post("2026-10-06T19:00:00+05:30"), post("2026-10-06T20:30:00+05:30")]
        now = datetime(2026, 10, 6, 8, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, day, []), datetime(2026, 10, 7, 17, 30, tzinfo=IST))

    def test_sent_posts_count_but_failed_and_drafts_do_not(self):
        rows = [post("2026-10-06T17:30:00+05:30", "sent"), post("2026-10-06T21:45:00+05:30", "error"),
                post("2026-10-06T19:00:00+05:30", "draft")]
        now = datetime(2026, 10, 6, 18, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, rows, []), datetime(2026, 10, 6, 21, 45, tzinfo=IST))

    def test_lead_time_skips_a_slot_that_is_too_close(self):
        now = datetime(2026, 10, 6, 17, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], []), datetime(2026, 10, 6, 21, 45, tzinfo=IST))

    def test_collision_within_an_hour_on_instagram(self):
        ig = [post("2026-10-06T17:00:00+05:30")]
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], ig), datetime(2026, 10, 6, 21, 45, tzinfo=IST))

    def test_no_slot_beyond_tomorrow(self):
        full = [post(f"2026-10-0{d}T19:00:00+05:30") for d in (6, 7)] + \
               [post(f"2026-10-0{d}T20:30:00+05:30") for d in (6, 7)]
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertIsNone(slots.free_slot(now, full, []))

    def test_lane_books_today_only(self):
        day = [post("2026-10-06T19:00:00+05:30"), post("2026-10-06T20:30:00+05:30")]
        now = datetime(2026, 10, 6, 8, 0, tzinfo=IST)
        self.assertIsNone(slots.free_slot(now, day, [], days_ahead=slots.BOOK_AHEAD_DAYS))
        late = datetime(2026, 10, 6, 21, 30, tzinfo=IST)
        self.assertIsNone(slots.free_slot(late, [], [], days_ahead=slots.BOOK_AHEAD_DAYS))

    def test_utc_works_from_any_zone(self):
        now = datetime(2026, 10, 5, 18, 40, tzinfo=timezone.utc)  # 00:10 IST on 6 October
        self.assertEqual(slots.utc(slots.free_slot(now, [], [])), "2026-10-06T12:00:00+00:00")

    def test_lite_slots_never_equal_signed_lane_slots(self):
        from datetime import time
        self.assertTrue(set(slots.LITE_SLOTS).isdisjoint({time(19, 0), time(20, 30), time(8, 30)}))


if __name__ == "__main__":
    unittest.main()
