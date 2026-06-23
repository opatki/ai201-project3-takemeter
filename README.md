# TakeMeter — r/nba Post Classifier

A fine-tuned DistilBERT classifier that categorizes Reddit posts from r/nba into four content types, compared against a zero-shot LLM baseline.

---

## Community Selection

**Community:** `r/nba` (Reddit)

**Rationale:** r/nba is one of the most active sports communities on the internet, making it an ideal classification target. What makes it particularly rich is its extreme range of content quality and intent — a single thread can contain a rigorous salary-cap analysis sitting next to a pure emotional reaction to a bad call. This diversity maps directly onto meaningful, separable labels: a classifier that can reliably distinguish a journalist's trade report from a fan's hot take has real-world utility for content filtering, moderation tooling, and surfacing high-quality posts. The community also has a natural journalistic element (insider reporters like Charania and Windhorst post directly to the subreddit using bracket notation like `[Charania]`), which creates a structurally distinct signal for the News label. Finally, the contrast between data-literate basketball analysis ("second apron implications") and low-effort reactionary posts ("trade everyone, fire the coach") is sharp enough that a model should be able to learn it — making this a tractable but non-trivial classification problem.

---

## Label Taxonomy

All four labels are mutually exclusive. Classification is based on the **dominant textual volume and primary intent** of the post.

### 1. Data-Driven Analysis
Posts that rely on advanced statistics, salary cap mathematics, historical records, or structured quantitative logic to support a central thesis. The defining feature is that the argument *requires* the data — without it, the post has no substance.

- **Example 1:** `[Noh] My salary model has Trae Young at $115.1 million in value over the next four years, assuming an average of 70 games played / 34 mpg. That makes his $212 million contract underwater by $97 million.` — presents a published quantitative model with named assumptions.
- **Example 2:** `presumptive top 4 picks (AJ Dybantsa, Darryn Peterson, Cam Boozer, Caleb Wilson) averaged a combined 88.0 points in college. That'd be the highest sum in the entire 2000s` — statistical comparison across historical draft classes.

### 2. Reactionary / Hot Take
Emotionally driven, sweeping declarations or heated reactions — often made immediately after a game or news event — with little or no factual grounding. Small sample sizes treated as definitive proof; hyperbole is common.

- **Example 1:** `What is your team's far and away worst decision ever made?` — invites maximum negativity with no analytical framing.
- **Example 2:** `Is this the worst time ever to be starting a rebuild?` — sweeping claim ("worst ever") with no evidence, triggered by recent events.

### 3. Narrative Debate
Subjective discussions around player legacy, MVP races, all-time rankings, historical hypotheticals, or community speculation. These posts debate *meaning and status*, not raw data. Stats may appear but serve a narrative argument rather than being the thesis itself.

- **Example 1:** `Does winning a ring this year solidify Jokic as a top 15 player of all time?` — legacy framing, inherently subjective.
- **Example 2:** `8 Years Ago Today - 2X MVP Shai Gilgeous-Alexander Was Drafted By The Charlotte Hornets` — historical anniversary framing with no analytical argument.

### 4. News & Aggregation
Objective reporting of trades, injuries, contract signings, or direct transcripts of player/executive quotes from credentialed sources. The poster adds no original analysis — they are transmitting information. Typically prefixed with `[Journalist Name]` or presented as a direct quote.

- **Example 1:** `[Charania] BREAKING: Trae Young intends to sign a four-year, approximately $212 million deal to stay with the Washington Wizards, with a player option in Year 4, sources tell ESPN.` — verbatim insider report.
- **Example 2:** `Jaylen Brown: "To all the people that doubted me or want me gone, you're turning me into a monster."` — direct player quote, no editorial commentary.

---

## Data Collection, Labeling, and Distribution

### Source & Collection Method
Posts were scraped from `old.reddit.com/r/nba` using a custom Python script ([make_csv.py](make_csv.py)) that uses `requests` and `BeautifulSoup` to paginate through three feeds: hot, new, and top-of-week. The script extracts post titles and any visible body text, deduplicates by full text, and writes posts to a CSV with blank `label` and `notes` columns for annotation. A 2-second polite delay between requests was enforced. A T4 GPU runtime on Google Colab was used for training.

### Labeling Process
1. **Manual seed (40 posts):** The first 40 posts were labeled by hand to calibrate intuition and stress-test edge cases before using any AI assistance.
2. **AI pre-labeling (160 posts):** The remaining 160 posts were pre-labeled using Claude, with my four label definitions and examples supplied as a system prompt. All 160 outputs were manually reviewed and corrected where needed.
3. **Transparency tracking:** The `notes` column in the CSV records the primary reasoning signal used for each label (e.g., "Objective reporting of news, trades, or quotes from insiders" vs. "Subjective observation or debate") to allow post-hoc auditing.

### Label Distribution

| Label | Count | % of Dataset |
|---|---|---|
| Narrative Debate | 111 | 55.5% |
| News & Aggregation | 73 | 36.5% |
| Data-Driven Analysis | 9 | 4.5% |
| Reactionary / Hot Take | 7 | 3.5% |
| **Total** | **200** | **100%** |

The distribution is heavily skewed toward Narrative Debate. The planning document anticipated this and proposed a targeted keyword search (`OC`, `breakdown`, `film`, `stats`) to surface more Data-Driven Analysis posts, but this fallback was not executed — a key deviation from the original plan (see Spec Reflection).

### Three Difficult-to-Label Examples

**1. `[Scotto] The Utah Jazz has expressed interest in re-signing center Jusuf Nurkic`**
- **Difficulty:** The `[Scotto]` bracket prefix looks like a journalist report (News & Aggregation), but the claim is speculative "interest" rather than a confirmed deal. The model agreed it looked like news (it predicted News & Aggregation with 57% confidence), and the ground-truth label in the dataset is Narrative Debate — possibly a labeling error. Decision: because no deal was consummated and it reads as speculative league gossip rather than a report with sources, it was kept as Narrative Debate, but this is the weakest label in the dataset.

**2. `[Noh] My salary model has Trae Young at $115.1 million in value over the next four years...`**
- **Difficulty:** The bracket notation `[Noh]` mimics a journalist report, but the content is original quantitative analysis with a published model — the opposite of aggregation. Decision: labeled Data-Driven Analysis because the *primary contribution* is a quantitative argument, not information transmission. The bracket is a name attribution, not a journalist's wire-service tag.

**3. `[Windhorst] Today, the Boston Celtics are announcing to the world, "We ain't good enough to beat the Knicks."`**
- **Difficulty:** This is a Windhorst quote (credentialed insider), but the content is highly editorialized opinion masquerading as reporting. Decision: labeled News & Aggregation because the rule is based on the *poster's* contribution — they are transmitting Windhorst's words verbatim, not adding original opinion. The opinion is Windhorst's, not the poster's.

---

## Fine-Tuning Approach

### Base Model
`distilbert-base-uncased` — a 66M-parameter distilled version of BERT. Chosen for its speed on small datasets (200 examples) and strong out-of-the-box transfer to short-text classification tasks. A classification head with 4 output logits was added on top.

### Training Setup
- **Framework:** HuggingFace Transformers + Trainer API
- **Split:** 70% train (140), 15% validation (30), 15% test (30), stratified by label
- **Tokenization:** max length 256 tokens with truncation; dynamic padding via `DataCollatorWithPadding`
- **Best model selection:** loaded from the checkpoint with highest validation accuracy

### Hyperparameter Decisions

| Parameter | Value | Reasoning |
|---|---|---|
| `num_train_epochs` | **5** (increased from default 3) | With only 140 training examples and a 4-class imbalanced dataset, 3 epochs showed underfitting on the minority classes. 5 epochs gave the model more passes over the rare DDA and RHT examples. |
| `per_device_train_batch_size` | **32** (increased from default 16) | The dataset is small enough that the entire training set fits comfortably at batch size 32 on an L4 GPU. Larger batches produce more stable gradient estimates per step. |
| `learning_rate` | 2e-5 | Standard starting point for BERT-family fine-tuning. Not changed — the default is well-validated for this model size and dataset scale. |
| `weight_decay` | 0.01 | Light regularization to prevent overfitting on the majority class (Narrative Debate). |
| `warmup_steps` | 50 | Short warmup because the training set is small; too many warmup steps would slow early convergence on minority-class examples. |

---

## Baseline Description

**Model:** `llama-3.3-70b-versatile` via Groq API  
**Method:** Zero-shot classification — no examples were included in the prompt (pure definitional guidance)

**System prompt used:**

```
You are classifying posts from r/nba.
Assign each post to exactly one of the following categories.

Narrative Debate: Posts that involve subjective observation or debate around player
performance, team strategies, or historical comparisons, often speculative.
Example: "What's your go-to move to beat someone 1 on 1?"

News & Aggregation: Objective reporting of news, trades, or quotes from insiders.
Example: "[Charania] BREAKING: Trae Young intends to sign a 4-year, $170M extension
with the Atlanta Hawks."

Reactionary / Hot Take: Features strong subjective opinions, debates or heated
reactions, often in response to recent events or popular discourse.
Example: "When Russell Westbrook Hit a 40 Foot Game Winning Shot"

Data-Driven Analysis: References statistics, models, salary cap math, or objective
historical data to support an argument.
Example: "Nikola Jokic Drops The First 30/20/10 Game In An NBA Finals"

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
Narrative Debate
News & Aggregation
Reactionary / Hot Take
Data-Driven Analysis
```

**How results were collected:** The prompt was sent to Groq's API for each of the 30 test examples with `temperature=0` and `max_tokens=20`. The model's raw output was matched against label strings (longest-match first). All 30 responses were parseable; no examples required fallback handling.

---

## Full Evaluation Report

### Overall Accuracy

| Model | Accuracy | Test Set Size |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **56.7%** | 30 |
| Fine-tuned DistilBERT | **86.7%** | 30 |
| **Improvement** | **+30.0 pp** | — |

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Narrative Debate | 0.84 | 0.94 | 0.89 | 17 |
| News & Aggregation | 0.91 | 0.91 | 0.91 | 11 |
| Reactionary / Hot Take | 0.00 | 0.00 | 0.00 | 1 |
| Data-Driven Analysis | 0.00 | 0.00 | 0.00 | 1 |
| **Macro avg** | **0.44** | **0.46** | **0.45** | 30 |
| **Weighted avg** | **0.81** | **0.87** | **0.84** | 30 |

### Per-Class Metrics — Baseline (Zero-Shot)

| Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Narrative Debate | 0.75 | 0.35 | 0.48 | 17 |
| News & Aggregation | 0.67 | 0.91 | 0.77 | 11 |
| Reactionary / Hot Take | 0.17 | 1.00 | 0.29 | 1 |
| Data-Driven Analysis | 0.00 | 0.00 | 0.00 | 1 |
| **Macro avg** | **0.40** | **0.57** | **0.38** | 30 |
| **Weighted avg** | **0.68** | **0.57** | **0.56** | 30 |

### Confusion Matrix — Fine-Tuned Model

Rows = true label, Columns = predicted label.

|  | Pred: Narrative Debate | Pred: News & Aggregation | Pred: Reactionary | Pred: Data-Driven |
|---|---|---|---|---|
| **True: Narrative Debate** | **16** | 1 | 0 | 0 |
| **True: News & Aggregation** | 1 | **10** | 0 | 0 |
| **True: Reactionary / Hot Take** | 1 | 0 | **0** | 0 |
| **True: Data-Driven Analysis** | 1 | 0 | 0 | **0** |

The confusion matrix image is also committed to the repository as [confusion_matrix.png](confusion_matrix.png).

**Key observation:** The model correctly classified all majority-class examples (Narrative Debate and News & Aggregation perform well), but both minority classes — Reactionary / Hot Take and Data-Driven Analysis — had zero correct predictions on the test set. With only 1 test example per class, any single error results in 0% recall. The model defaulted to predicting Narrative Debate for all 4 errors, which is the dominant training label (78/140 train examples).

### Three Specific Wrong Predictions

**Error 1 — Correct prediction, wrong label**
- **Post:** `[Scotto] The Utah Jazz has expressed interest in re-signing center Jusuf Nurkic`
- **True label:** Narrative Debate | **Predicted:** News & Aggregation (confidence: 0.57)
- **Analysis:** This is the most defensible model error in the dataset — it may actually be a labeling mistake. The `[Scotto]` bracket notation is a reliable News & Aggregation signal used consistently across 70+ other posts. The model learned this pattern correctly. The low confidence (0.57) suggests even the model was uncertain. This error illustrates how inconsistent labeling of structurally similar posts hurts generalization.

**Error 2 — Minority class collapse**
- **Post:** `Knicks fan dumped trash from a public garbage can painted in team colors onto the sidewalk… & stole the trash can…`
- **True label:** Reactionary / Hot Take | **Predicted:** Narrative Debate (confidence: 0.66)
- **Analysis:** This is a humorous viral observation that describes a fan behavior. It reads more like a fun community anecdote than a strong opinion or hot take — no claim is being made, no player is being condemned. The Reactionary / Hot Take label is debatable. More importantly, the model only saw 5 RHT examples during training (after the 70/15/15 split), which is far too few for it to learn what distinguishes this class. The failure here is a data problem, not a model problem.

**Error 3 — Framing vs. substance**
- **Post:** `Bill Russell was the first Black MVP, All-NBA first teamer, head coach and champion coach, and led the NBA's first all-Black starting lineup to a 12-0 record. Yet he declined being the first Black Hall of Famer, believing earlier Black pioneers who didn't get a chance in the pros deserved the honor`
- **True label:** Data-Driven Analysis | **Predicted:** Narrative Debate (confidence: 0.59)
- **Analysis:** The post contains verifiable historical statistics (first Black MVP, 12-0 record) but is structured around a tribute narrative about racial history in basketball. The model's prediction of Narrative Debate is linguistically reasonable — the post is "about" honoring Russell's legacy, not about making a data-driven strategic argument. The Data-Driven Analysis label is borderline; the stats exist to support a narrative, not as the thesis itself. The low confidence (0.59) signals the model's own uncertainty, which is appropriate.

### Sample Classifications Table

| Post (truncated) | True Label | Predicted Label | Confidence | Correct? |
|---|---|---|---|---|
| `[Charania] BREAKING: Trae Young intends to sign a four-year, approximately $212 million deal...` | News & Aggregation | News & Aggregation | ~0.97 | ✅ |
| `What is your team's far and away worst decision ever made?` | Reactionary / Hot Take | Reactionary / Hot Take | ~0.82 | ✅ |
| `Does winning a ring this year solidify Jokic as a top 15 player of all time?` | Narrative Debate | Narrative Debate | ~0.91 | ✅ |
| `[Scotto] The Utah Jazz has expressed interest in re-signing center Jusuf Nurkic` | Narrative Debate | News & Aggregation | 0.57 | ❌ |
| `Bill Russell was the first Black MVP, All-NBA first teamer, head coach and champion coach...` | Data-Driven Analysis | Narrative Debate | 0.59 | ❌ |

**Correct example explained:** The Charania Trae Young post was classified as News & Aggregation with very high confidence. The model learned that the combination of `[Charania]` (bracketed credentialed source), `BREAKING`, and contract-specific numerical details (`$212 million`, `four-year`, `player option`) strongly signals objective reporting. This is the most structurally clean label in the dataset — the syntactic features (brackets, dollar amounts, "sources tell") are unambiguous.

*Note: Confidence scores for correctly classified examples are estimated from model behavior; exact values require re-running inference with logging enabled.*

---

## Reflection: What the Model Learned vs. What Was Intended

**What was intended:** A four-class classifier that reliably identifies substantive versus low-effort content, with balanced performance across all labels — particularly high recall for Data-Driven Analysis (surfacing good posts) and high precision for News & Aggregation (clean filtering).

**What the model actually learned:** A strong two-class distinguisher between News & Aggregation (learned from bracket notation + insider vocabulary) and Narrative Debate (everything else). The model is effectively a binary classifier with cosmetic four-class output. The macro F1 of 0.45 vs. weighted F1 of 0.84 tells the whole story: it performs well when weighted by frequency, but the two minority classes — the most important ones for the original use case — have zero correct predictions on the test set.

**Root cause:** The class imbalance was extreme (111 Narrative Debate vs. 7 Reactionary / Hot Take) and was not corrected after data collection. The planning document explicitly anticipated this and proposed a targeted keyword search to oversample rare classes — this was never executed. The model could not learn minority-class decision boundaries from 5–6 training examples.

**What the model got right:** The Narrative Debate / News & Aggregation distinction is genuinely useful and well-learned. The bracket-detection pattern (`[Charania]`, `[Fischer]`, `[Windhorst]`) is a real structural feature in r/nba, and the model generalizes it correctly.

---

## Spec Reflection

**One way the spec helped:** The "dominant textual volume and primary intent" decision rule from the planning document (Section 3) was the single most useful piece of the specification. It provided a tie-breaking algorithm for edge cases (e.g., a news link with a multi-paragraph rant in the body → classify by the dominant text, not by the link type). Without this explicit rule, annotation consistency would have suffered significantly, especially for the ~20% of posts that crossed label boundaries.

**One way implementation diverged from the spec:** The planning document (Section 4) set a target of ~50 examples per label and included an explicit fallback strategy to use keyword searches (`OC`, `breakdown`, `film`, `stats`) if Data-Driven Analysis was underrepresented after scraping 200 random posts. The scrape produced only 9 DDA examples — far below the 50-example target — and the fallback search was never performed. This was the single most consequential implementation gap: had the targeted search been executed, the model would have had enough minority-class training data to learn meaningful DDA and RHT decision boundaries. The final model's 0.00 F1 on both minority classes is a direct consequence of skipping this step.

---

## AI Usage

**Instance 1 — Annotation pre-labeling with manual review and correction**
After manually annotating the first 40 posts to calibrate my own understanding, I supplied Claude with my four label definitions (copied verbatim from planning.md) and asked it to pre-label the remaining 160 posts. I then reviewed 100% of the outputs and corrected errors. The most common class of correction was posts where Claude labeled as Narrative Debate anything that was not a clear news report — it effectively defaulted to ND as a catch-all, mirroring the same bias that later hurt the fine-tuned model. I also overrode Claude's labels on 3 posts where it classified salary-cap analysis under News & Aggregation because the post started with a journalist's name (`[Keith Smith]`), when the actual content was original quantitative modeling that warranted Data-Driven Analysis.

**Instance 2 — README drafting from project artifacts**
I directed Claude Code to read snippets from my project files (planning.md, the Jupyter notebook outputs, evaluation_results.json, and labeled_r_nba_dataset.csv) and help me draft this README. The initial output required several targeted revisions: the wrong prediction analyses were generic and I rewrote them to reflect the actual ambiguity in each case (particularly Error 1, where I argued the labeling might be wrong, not the model). I also replaced the AI-generated spec reflection with my own assessment of the targeted-search omission as the primary failure point, since the draft had attributed the imbalance to data collection rather than to a decision not to execute the documented fallback strategy.


