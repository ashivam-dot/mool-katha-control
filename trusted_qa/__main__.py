"""Run `python -m trusted_qa` from a future trusted, private QA repository."""

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
