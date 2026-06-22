# TakeMeter — planning.md

---

## 1. Community Selection
**Community:** `r/nba` (Reddit)
**Why it's a good fit:** I chose `r/nba` because it is one of the most active sports forums on the internet, featuring a massive daily volume of text-heavy posts and comments. It is a perfect fit for a classification task because the discourse varies wildly in quality and intent. Within a single thread, you can find brilliant statistical breakdowns of complex defensive schemes right next to purely emotional, reactionary attacks on a player's character. Categorizing this spectrum from "substantive" to "low-effort" is highly relevant for community moderation and content filtering.

## 2. Label Definitions
Based on the "Substance & Effort" framework, I will use four mutually exclusive labels:

* **Data-Driven Analysis:** Posts that rely on advanced statistics, film breakdown, salary cap mathematics, or structural logic to support a central thesis.
    * *Example 1:* "Why the Timberwolves' drop coverage is statistically elite against PnR guards (Film + Synergy Stats)"
    * *Example 2:* "Breaking down the new CBA: Why the Celtics cannot afford a 3rd max contract under the second apron."
* **Reactionary / Hot Take:** Emotionally driven, sweeping declarations or conclusions based on extremely small sample sizes (often right after a game), lacking factual or statistical backing.
    * *Example 1:* "Player X is completely washed and this franchise is doomed for the next decade."
    * *Example 2:* "Trade everyone. Build around the rookie. The coach needs to be fired tonight."
* **Narrative Debate:** Discussions focused on subjective legacy, MVP races, all-time player rankings, or historical hypotheticals rather than on-court X's and O's.
    * *Example 1:* "Does winning a ring this year solidify Jokic as a top 15 player of all time?"
    * *Example 2:* "Who had the harder path to the Finals: the 2011 Mavericks or the 1995 Rockets?"
* **News & Aggregation:** Objective reporting of trades, injuries, Wojnarowski/Shams tweets, or direct transcripts of player interviews with no added original analysis.
    * *Example 1:* "[Wojnarowski] The Lakers are trading for..."
    * *Example 2:* "Giannis Antetokounmpo will be out for 2-4 weeks with a calf strain."

## 3. Hard Edge Cases
* **The Ambiguous Case:** A post that starts as **News** (e.g., a link to a trade tweet) but the user includes a three-paragraph emotional rant in the post body about why the trade ruins the team (**Reactionary**). Another edge case is using stats (Analysis) to argue an MVP case (Narrative Debate).
* **Handling Strategy:** I will classify based on the *dominant textual volume and primary intent* of the user's contribution. If a post is a news link but contains >50% original analytical text, it becomes "Data-Driven Analysis." If stats are used purely to construct an all-time ranking, it falls under "Narrative Debate" because the *intent* is legacy discussion. If the text is purely emotional venting over a news link, it is "Reactionary."

## 4. Data Collection Plan
* **Sources:** I will collect examples using the Reddit API (PRAW) or manual scraping, targeting top "Text" posts from the past week, as well as highly upvoted/downvoted comments inside "Post-Game Threads" to capture the "Hot Take" element.
* **Target Size:** 200 total examples, aiming for an even split (~50 per label). 
* **Underrepresentation Fallback:** "Data-Driven Analysis" is generally much rarer than "Reactionary" posts. If Analysis is underrepresented after scraping 200 random posts, I will do a targeted search using keywords like "OC" (Original Content), "breakdown," "stats," or "film" to pull 20–30 specific examples to balance the dataset.

## 5. Evaluation Metrics
* **Primary Metrics:** Macro F1-Score, Precision, and Recall.
* **Why Accuracy isn't enough:** The classes will likely be naturally imbalanced (Hot Takes vastly outnumber Data-Driven Analysis). If a model just guesses "Reactionary" every time, it might achieve high accuracy but fail entirely at its purpose. 
* **Why Precision/Recall matter:** Precision is crucial for the "News" label—we absolutely do not want opinion pieces masquerading as factual news. Recall is vital for "Data-Driven Analysis" so that high-effort, substantive posts aren't accidentally filtered out. F1-Score will help balance these competing needs across all four classes.

## 6. Definition of Success
* **Useful Performance:** A baseline Macro F1-score of > 0.75 would demonstrate the model actually understands the distinct linguistic patterns of the classes.
* **"Good Enough" for Deployment:** To deploy this as a real community tool (e.g., a Reddit filter extension), it must achieve:
    1.  **> 85% Precision on "News & Aggregation"** (Ensuring users who filter for "News Only" don't get junk opinions).
    2.  **> 75% Recall on "Data-Driven Analysis"** (Ensuring the tool successfully identifies and surfaces high-effort content without burying it).

## 7. AI Tool Plan
* **Label Stress-Testing:** Before I begin my 200 annotations, I will prompt an LLM (like Claude or ChatGPT) with my label definitions and ask it to generate 10 "edge case" NBA posts that deliberately sit on the boundary between labels (e.g., Narrative vs. Analysis). If I struggle to classify the LLM's examples, I will refine my definitions before annotating the real data.
* **Annotation Assistance:** I will manually annotate the first 40 posts to ground my understanding. I will then use an LLM (via API or prompt) to pre-label the remaining 160 posts. I will manually review 100% of these pre-labeled outputs, correcting any mistakes. I will track which labels were changed in an "LLM_Corrected" column in my dataset for transparency.
* **Failure Analysis:** After training and testing my classifier, I will aggregate all False Positives and False Negatives. I will feed these errors into an LLM and prompt it to "identify linguistic or structural patterns among these misclassified posts." I will manually review the LLM's theories to see if the model over-indexed on specific player names, profanity, or post length rather than substance.
