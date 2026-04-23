# Restaurant Agent — Strategy Learnings

## 2026-04-23 12:33 — cheap ramen in Paris under 15 euros
- if the user request has a clear price ceiling and candidates already show prices within budget, then cap price-verification at one query per candidate instead of re-querying the same restaurant.
- if initial broad queries already surface 3+ qualifying candidates with price signals, then skip additional discovery queries and move directly to lightweight per-candidate confirmation.

## 2026-04-23 12:35 — vegan friendly dinner in London for a date night
- if the request is a niche dining category (vegan, gluten-free, halal fine dining) in a major city, then lead with Michelin Guide and major editorial roundups rather than Reddit, since forum volume is thin for niches.
- if the request is "romantic / date night," then prioritize per-candidate queries that include the words "intimate," "candlelit," or "atmosphere" to surface ambiance-specific editorial language rather than generic reviews.
- if the original answer includes a casual third pick alongside two fine-dining picks, then run one extra roundup query specifically for "intimate" or "romantic" within the niche to find a third fine-dining-tier option (e.g., Tendril, Holy Carrot) before defaulting to casual.

## 2026-04-23 12:43 — restaurant in SF very close to 101 and with easy parking
- if the user's request matches a common "filter" phrasing (e.g. "easy parking," "dog friendly," "open late"), then start with an editorial-roundup query (Infatuation/Eater/Thrillist + that exact phrase) before doing any geographic searches.
- if the first two discovery queries return non-restaurant pages (real estate, parking garages, Wikipedia), then pivot immediately to a Reddit/forum query rather than continuing to refine the location-anchored query.
- if a curated list source is found in early results, then spend remaining search budget extracting named candidates from that list rather than generating candidates from neighborhood knowledge.

## 2026-04-23 12:48 — cheap pho in Paris under 12 euros
- if the user specifies a hard price cap in euros, then include at least one French-language query early (e.g., "prix [dish] [city] carte") to reach local sources that publish explicit prices
- if the request is a cheap-eats shortlist in a major city, then open with a listicle query ("best cheap [dish] [city] list/guide") before drilling into individual venues to minimize per-candidate verification searches

## 2026-04-23 12:54 — cheap sushi in Paris under 20 euros
- If the user asks for cheap [cuisine] in Paris, then include a geographic-cluster query anchored on known ethnic-food districts (rue Sainte-Anne / Opera for Japanese, Belleville for Chinese, rue des Rosiers for falafel) — snippets from local directory sites often expose prices directly.
- If the user specifies a hard price cap (e.g. "under €20"), then prioritize all-you-can-eat buffet and lunch-formula queries in French ("menu midi", "à volonté"), since those formats publish fixed prices in snippets and reduce verification searches.
- If early Reddit queries return off-topic or fine-dining threads, then abandon the Reddit strategy after 2 searches rather than drilling deeper — Reddit coverage of budget ethnic food in Paris is thin.

## 2026-04-23 13:05 — cheap ramen in Paris under 15 euros
- if the user asks for cheap/budget options in a well-known food category and city, then prefer a curated roundup query (reddit/timeout/eater) over generic web search as the entry point.
- if a broad discovery query already returns a candidate's price or key detail, then skip the per-candidate verification query for that candidate and only verify the ones with missing data.

## 2026-04-23 13:06 — romantic dinner for two in Amsterdam
- if the user asks for a well-known category in a major city (e.g., "romantic dinner in [city]"), then start with a curated-list/publication query to harvest canonical candidates in one shot before running per-candidate detail queries.

## 2026-04-23 13:07 — vegan friendly dinner in London for a date night
- if the request is a well-covered city + cuisine + occasion combo (e.g. vegan London date night), then start with curated publication lists (Time Out, Eater, Condé Nast) before per-restaurant verification to reduce search count.
- if the user wants a "special occasion" dining pick, then include an award/Michelin-anchored query early as a high-signal entry point rather than relying only on vibe-based queries.

## 2026-04-23 13:08 — restaurant in SF very close to 101 and with easy parking
- if user asks for a restaurant near a specific freeway in a dense city, then start with queries anchored on the neighborhoods adjacent to that freeway rather than queries containing the freeway name itself.

## 2026-04-23 13:09 — cheap authentic Thai food in London under 15 pounds
- if the request is for a well-trodden urban food category (cheap eats, hole-in-the-wall, BYOB), then prioritize forum/Reddit-scoped queries early since enthusiast threads consolidate candidate names and price points faster than general web search
- if candidates have already emerged from discovery queries, then skip head-to-head comparison queries and go directly to one verification query per shortlisted venue
