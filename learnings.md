# Restaurant Agent — Strategy Learnings


## 2026-04-23 13:58 — cheap ramen in Paris under 15 euros
- if the user request is bounded by both price and city, then start with a map/listing-style query (Google Maps, TimeOut, Reddit roundup) before per-restaurant verification to collect multiple priced candidates in one search.  [u:0 h:0]
- if early queries converge on a known restaurant cluster (e.g., a single street), then cap per-restaurant verification at one query each rather than re-querying the same venue.  [u:0 h:0]
