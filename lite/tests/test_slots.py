import unittest
from datetime import datetime, timezone

from lite import slots

IST = slots.IST


def post(when: str, status: str = "scheduled") -> dict:
    return {"id": when, "status": status, "dueAt": datetime.fromisoformat(when).isoformat()}


class FreeSlotTest(unittest.TestCase):
    def test_first_lite_slot_today(self):
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], []), datetime(2026, 10, 6, 12, 30, tzinfo=IST))

    def test_booked_slots_are_skipped_in_order(self):
        now = datetime(2026, 10, 6, 0, 10, tzinfo=IST)
        booked = []
        for expected in ((6, 12, 30), (6, 17, 30), (6, 21, 45), (7, 12, 30)):
            got = slots.free_slot(now, booked, booked)
            self.assertEqual(got, datetime(2026, 10, expected[0], expected[1], expected[2], tzinfo=IST))
            booked.append(post(got.isoformat()))

    def test_three_shorts_a_day(self):
        day = [post("2026-10-06T10:00:00+05:30"), post("2026-10-06T15:00:00+05:30"), post("2026-10-06T20:00:00+05:30")]
        now = datetime(2026, 10, 6, 8, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, day, []), datetime(2026, 10, 7, 12, 30, tzinfo=IST))

    def test_sent_posts_count_but_failed_and_drafts_do_not(self):
        rows = [post("2026-10-06T17:30:00+05:30", "sent"), post("2026-10-06T21:45:00+05:30", "error"),
                post("2026-10-06T19:00:00+05:30", "draft")]
        now = datetime(2026, 10, 6, 18, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, rows, []), datetime(2026, 10, 6, 21, 45, tzinfo=IST))

    def test_lead_time_skips_a_slot_that_is_too_close(self):
        now = datetime(2026, 10, 6, 17, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], []), datetime(2026, 10, 6, 21, 45, tzinfo=IST))

    def test_collision_within_an_hour_on_instagram(self):
        ig = [post("2026-10-06T12:00:00+05:30")]
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertEqual(slots.free_slot(now, [], ig), datetime(2026, 10, 6, 17, 30, tzinfo=IST))

    def test_no_slot_beyond_tomorrow(self):
        full = [post(f"2026-10-0{d}T{t}:00+05:30") for d in (6, 7) for t in ("12:30", "17:30", "21:45")]
        now = datetime(2026, 10, 6, 9, 0, tzinfo=IST)
        self.assertIsNone(slots.free_slot(now, full, []))

    def test_lane_books_today_only(self):
        day = [post(f"2026-10-06T{t}:00+05:30") for t in ("12:30", "17:30", "21:45")]
        now = datetime(2026, 10, 6, 8, 0, tzinfo=IST)
        self.assertIsNone(slots.free_slot(now, day, [], days_ahead=slots.BOOK_AHEAD_DAYS))
        late = datetime(2026, 10, 6, 21, 30, tzinfo=IST)
        self.assertIsNone(slots.free_slot(late, [], [], days_ahead=slots.BOOK_AHEAD_DAYS))

    def test_utc_works_from_any_zone(self):
        now = datetime(2026, 10, 5, 18, 40, tzinfo=timezone.utc)  # 00:10 IST on 6 October
        self.assertEqual(slots.utc(slots.free_slot(now, [], [])), "2026-10-06T07:00:00+00:00")

    def test_slots_keep_the_gap_and_miss_the_story_time(self):
        from datetime import date, datetime as dt, time
        times = [dt.combine(date(2026, 10, 6), t) for t in slots.LITE_SLOTS]
        self.assertTrue(all(b - a >= slots.MIN_GAP for a, b in zip(times, times[1:])))
        self.assertNotIn(time(8, 30), slots.LITE_SLOTS)


if __name__ == "__main__":
    unittest.main()
