import requests
from bs4 import BeautifulSoup
import csv
import time

# Mimic a standard browser to avoid getting blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Base feeds to paginate through
base_urls = [
    'https://old.reddit.com/r/nba/',
    'https://old.reddit.com/r/nba/new/',
    'https://old.reddit.com/r/nba/top/?sort=top&t=week'
]

TARGET = 200
posts = []
seen_texts = set()

def scrape_page(url):
    print(f"Scraping {url}...")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"  Skipping: status {response.status_code}")
        time.sleep(5)
        return [], None

    soup = BeautifulSoup(response.text, 'html.parser')
    found = []

    for thing in soup.find_all('div', class_='thing'):
        title_elem = thing.find('a', class_='title')
        if not title_elem:
            continue

        title = title_elem.text.strip()

        # Include body text for self posts if visible
        expando = thing.find('div', class_='usertext-body')
        body = expando.text.strip() if expando else ""

        text = f"{title} {body}".strip()

        if len(text) > 20 and text not in seen_texts:
            seen_texts.add(text)
            found.append({'text': text})

    # Find the "next page" link
    next_button = soup.find('span', class_='next-button')
    next_url = next_button.find('a')['href'] if next_button else None

    return found, next_url

for base_url in base_urls:
    if len(posts) >= TARGET:
        break

    current_url = base_url
    pages_fetched = 0

    while current_url and len(posts) < TARGET and pages_fetched < 10:
        new_posts, next_url = scrape_page(current_url)
        posts.extend(new_posts)
        print(f"  Found {len(new_posts)} new posts (total: {len(posts)})")

        if not next_url or not new_posts:
            break

        current_url = next_url
        pages_fetched += 1
        time.sleep(2)  # Be polite to Reddit's servers

# Save to CSV
with open('r_nba_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label', 'notes'])  # Required columns
    for post in posts[:TARGET]:
        writer.writerow([post['text'].replace('\n', ' '), '', ''])

print(f"Saved {min(len(posts), TARGET)} posts to r_nba_dataset.csv!")
