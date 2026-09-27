# Error Analysis Report

- **Validation Pairs Evaluated**: 300,000
- **True Positives (TP)**: 1,586
- **False Negatives (FN)**: 218
- **False Positives (FP)**: 195
- **Decision Threshold**: 0.4500

## Top 5 Weakest Features in False Negatives (Missed Matches)

| Feature | FN Mean | TP Mean | Drop in FN |
| :--- | :--- | :--- | :--- |
| `addr_token_overlap` | 0.5249 | 0.8577 | +0.3328 |
| `addr_token_jaccard` | 0.3852 | 0.6961 | +0.3110 |
| `addr_tfidf_cosine` | 0.5534 | 0.8592 | +0.3058 |
| `addr_levenshtein` | 0.5524 | 0.8208 | +0.2685 |
| `addr_jaro_winkler` | 0.6031 | 0.8667 | +0.2636 |

## Top 5 Features Distinguishing False Positives from True Negatives

| Feature | FP Mean | TN Mean | Elevation in FP |
| :--- | :--- | :--- | :--- |
| `addr_tfidf_cosine` | 0.7777 | 0.0491 | +0.7287 |
| `addr_token_overlap` | 0.7689 | 0.0418 | +0.7271 |
| `addr_token_jaccard` | 0.5952 | 0.0204 | +0.5749 |
| `name_token_jaccard` | 0.6800 | 0.2222 | +0.4577 |
| `addr_levenshtein` | 0.7877 | 0.3373 | +0.4504 |

## Sample False Negatives (Lowest Confidence Missed True Matches)

| S1 Name | Cand Name | S1 Address | Cand Address | Prob | Name Jaccard | Addr Levenshtein |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| casillas lessard and rhames | casillas lessard and rhames in | hitchcock 6812 1st street tx | 5812 1nd saint hitchcock texas | 0.000 | 1.00 | 0.45 |
| abhikalp foundation limited | abhikalp f0undation limited 23 | 4 cauvery avenue alagapuram sa |  | 0.000 | 0.25 | 0.00 |
| delhi chem india | delhi chem services services | tower 2 svasa homes no 13 5th  |  | 0.001 | 0.67 | 0.00 |
| shree industries private limit | శ ర ఇ డస ట ర స ప ర వ ట ల మ ట డ | hno 15 74 tekulapalli khammam  | 15 74 tekulapalli త ల గ ణ kham | 0.001 | 0.00 | 0.69 |
| hotel consulting private limit | ഹ ട ടൽ കൺസൾട ട ഗ പ ര വറ റ ല മ  | c o noushad gulzar house govt  | c o noushad gulzar house triva | 0.001 | 0.00 | 0.57 |
| hotel consulting private limit | ഹ ട ടൽ കൺസൾട ട ഗ പ ര വറ റ ല മ  | c o noushad gulzar house govt  | c o noushad gulzar house triva | 0.001 | 0.00 | 0.57 |
| global assets incorporated | global asgmlss ínc | 5304 madison 4448 saint paul a | 5304 1 2 madison 4448 n a comb | 0.001 | 0.25 | 0.68 |
| peyton s mountain hospitality | peyton s mountain 78918 | 5401 groveside lane rolling me |  | 0.002 | 0.60 | 0.00 |
| vas cv private limited | vas private limited services | plot no 25 gayathri nagar srin |  | 0.002 | 0.50 | 0.00 |
| suvidha welfare society | sri suvidha welfare cénter | s 7 plot no4 pkt 7 sf malhan p |  | 0.003 | 0.40 | 0.00 |

## Sample False Positives (Highest Confidence False Matches)

| Flag | S1 Name | Cand Name | S1 Address | Cand Address | Prob |
| :--- | :--- | :--- | :--- | :--- | :--- |
| General FP | brightium robinhood llc | brightiem robdhood llc | 33269 190th street 22 ia | 22 33269 190th street ia | 1.000 |
| General FP | sb highland mines llc | sbf híghland mines | 25990 cedar valley road natron | 25990 cedar valley road natron | 0.996 |
| General FP | sturgill s great management | sturgill s great industries | 8708 whistling straits way kno | 8719 1 2 whistling straits way | 0.993 |
| General FP | ro infra newdelhi | so infra newdelhi | newdelhi delhi 4th floor zakir | b 655 21 4th floor zakir nagar | 0.988 |
| General FP | la polytechnic private limited | la solutions private limited | 1 construction housewalchand h | 4 construction housewalchand h | 0.988 |
| General FP | koga fuse incorporated | koga fuse incorporated | 4895 389 road stuart ok | 4908 389 road stuart oklahoma | 0.987 |
| General FP | barker health center incorpora | incorporated barker health cen | richmond heights oh 488 pierso | 490 pierson drive cleveland oh | 0.984 |
| General FP | bangalore steels private limit | bangalore msaets private limit | bangalore 6th block 22nd cross | karnataka t303 suraj ganga arc | 0.981 |
| General FP | phase v dlf city projects priv | phase v dlf cíty textiles priv | haryana the summit sub 123 pha | sub 127 the summit phase v dlf | 0.980 |
| General FP | imojean boyd signature china i | imojean boyd signature fashion | 29w 270 forest avenue west chi | 29w 273a forest avenue il west | 0.978 |
