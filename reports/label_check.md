# Hand-check of 30 MD labels (random sample of md_steps_n600, random_state=0)

Contact sheets: `reports/figs/md_label_check_01..10.png` (source screen with the matched OCR window boxed in red,
a native-resolution zoom of that window, and the current screen). Checked by inspecting the images
(Claude, 2026-10-02). Criterion: the needed string is (i) genuinely shown on an earlier screen of the same
episode and (ii) not visible on the current screen, i.e. the agent must remember it.

| # | episode / step | needed string | verdict | note |
|---|---|---|---|---|
| 1 | 1489856008258325/9 | GoPro HERO10 | ✓ | product title on step 5; in instruction |
| 2 | 4636417508451352/6 | 150g salted butter, … (ingredient list) | ✓ | full ingredient list on recipe page |
| 3 | 2803993314796469/17 | french language class | ✓ | Tripadvisor title; in instruction |
| 4 | 3678335785303445/11 | new delhi | ✓ | earlier search query; in instruction |
| 5 | 9249888902239790/8 | VR Development Full Course: Oculus Quest | ✓ | playlist title |
| 6 | 0193478855322542/12 | Holy Stone HS720R | ✓ | drone product title |
| 7 | 4310579014484848/8 | do yoga with this | ✓ (borderline) | string is the agent's own earlier typed note, visible on step 6 |
| 8 | 4174759152335380/24 | the shindellas | ✓ | artist name in article |
| 9 | 3335264302016436/17 | DealMoon | ✓ (borderline) | step 16 shows mistyped "DealMon"; in instruction |
| 10 | 2231862333326482/10 | Great R&B | ✓ | Spotify playlist |
| 11 | 3980360413822387/8 | Waze | ✓ | search result |
| 12 | 4264806779091821/8 | Google's AI Course for Beginners(in 10 minutes)! | ✓ | YouTube title |
| 13 | 7339822285663511/13 | THE OUTNET | ✓ | app name; in instruction |
| 14 | 3219448417872195/8 | Quinoa Bowl | ✓ | Quora answer list |
| 15 | 9464647543951273/13 | Citizen Science Fair will come on MAY 25 | ✗ | **visible on current screen** (white text in purple chat bubble; OCR missed it) |
| 16 | 1796008213248338/11 | Escondite | ✓ | destination in maps |
| 17 | 7233930811894516/15 | Goddess Bowls | ✓ | recipe title |
| 18 | 4389994022937346/12 | Yellowstone National Park | ✓ | post title; in instruction |
| 19 | 6125190889041336/2 | book about historical fiction | ✗ | match is stale browser history from a previous session (not episode memory); truncated tile also on current screen |
| 20 | 3538165452055539/8 | table tennis champion game | ✗ | fuzzy match to "Table Tennis Championships" (different string; typed text is a paraphrase) |
| 21 | 0180872754718232/13 | Lazada | ✓ | app name; in instruction |
| 22 | 7116642307811069/2 | Tokopedia | ✓ | home-screen icon label; in instruction |
| 23 | 8006965329628211/9 | Grilled Chicken Salad | ✓ | |
| 24 | 1061013841071298/12 | Los Angeles | ✓ | location of chosen trail |
| 25 | 0589394936591331/8 | sonet | ✓ | "Kia Sonet" found by search, then entered in model picker — clean memory case |
| 26 | 0800508265572461/5 | 640 San Julian St | ✓ | apartment address |
| 27 | 3434659461858971/26 | Bristlecone | ✓ | plant name, 4 steps earlier |
| 28 | 3810495333995541/7 | Waze | ✓ | |
| 29 | 8498065817312141/17 | porsche | ✓ | earlier search; in instruction |
| 30 | 3042657515534914/17 | spanish language class | ✓ | video title; in instruction |

**Precision: 27/30 = 0.90** (Wilson 95% CI ≈ 0.74–0.97). Two of the 27 are borderline-but-correct.
Failure modes: OCR misses light text on coloured backgrounds (→ "not on current screen" wrongly true);
stale device state (browser history) counted as an earlier screen; 0.8 fuzzy threshold admits near-paraphrases.
Of the 30, 13 needed strings are also in the instruction (not "strict" MD); 17 are strict.
