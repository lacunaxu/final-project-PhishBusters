import os
import time
from typing import List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_openphish_urls() -> List[str]:
    """
    Fetch a list of URLs from OpenPhish.

    Returns:
        List[str]: A list of phishing URLs from OpenPhish.
    """
    openphish_url = "https://openphish.com/feed.txt"
    try:
        response = requests.get(openphish_url, timeout=30)
        if response.status_code == 200:
            urls = response.text.splitlines()
            return urls
        else:
            print(f"Error fetching OpenPhish URLs, response code: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Error: {e}")
        return []


def fetch_tranco_urls(n: int = 500) -> List[str]:
    """
    Fetch the top `n` URLs from the Tranco list.

    Parameters:
        n (int): The number of top sites to fetch. Default is 500.

    Returns:
        List[str]: A list of URLs from the Tranco list.
    """
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(os.path.abspath("../data/raw/tranco_top_sites.csv.zip"), "wb") as f:
                f.write(response.content)
            df = pd.read_csv(
                os.path.abspath("../data/raw/tranco_top_sites.csv.zip"), header=None, names=["Rank", "Domain"]
            ).head(n)
            urls = []
            for domain in df["Domain"]:
                www_url = f"http://www.{domain}"
                try:
                    if requests.get(www_url, timeout=10).status_code == 200:
                        urls.append(www_url)
                    else:
                        urls.append(f"http://{domain}")
                except requests.RequestException:
                    urls.append(f"http://{domain}")
            return urls
        else:
            print(f"Failed to fetch Tranco list: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Tranco URLs: {e}")
        return []


def fetch_HTML_urls(url: str) -> Optional[str]:
    """
    Fetch the HTML content of a given URL.

    Parameters:
        url (str): The URL to fetch.

    Returns:
        Optional[str]: The HTML content as a string if successful, None otherwise.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except requests.RequestException:
        return None


def fetch_urls(url: str, max_urls: int = 10) -> List[str]:
    """
    Crawl all links on a given website up to a maximum number of URLs.

    Parameters:
        url (str): The seed URL to start crawling from.
        max_urls (int): The maximum number of URLs to fetch. Default is 10.

    Returns:
        List[str]: A list of extracted URLs.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    used = set()
    extracted = set()
    urls = {url}

    while urls and len(extracted) < max_urls:
        current_url = urls.pop()
        if current_url in used:
            continue
        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
        except requests.exceptions.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            if len(extracted) >= max_urls:
                break

            full_url = urljoin(current_url, a_tag["href"])
            if url in full_url and full_url not in used:
                urls.add(full_url)
                extracted.add(full_url)

        used.add(current_url)
        time.sleep(1)

    return list(extracted)


if __name__ == "__main__":
    print("Fetching data... This process may take over an hour.")

    """
    all_urls_output_file = os.path.abspath("../data/processed/all_urls.csv")
    tranco_output_file = os.path.abspath("../data/raw/tranco_urls.csv")
    openphish_file = os.path.abspath("../data/raw/openphish_urls.csv")

    openphish_urls = fetch_openphish_urls()
    tranco_urls = fetch_tranco_urls(n=500)
    print(f"Fetched {len(openphish_urls)} OpenPhish URLs")
    print(f"Fetched {len(tranco_urls)} Tranco URLs")

    openphish_urls = pd.DataFrame(openphish_urls, columns=["URL"])
    openphish_urls.to_csv(openphish_file, index=False)

    tranco_df = pd.DataFrame(tranco_urls, columns=["URL"])
    tranco_df.to_csv(tranco_output_file, index=False)
    print(f"Tranco URLs saved to {tranco_output_file}")

    seed_sites = list(tranco_df["URL"])
    all_urls = []
    for site in seed_sites:
        try:
            all_urls.extend(fetch_urls(site))
        except Exception as e:
            print(f"Error processing site {site}: {e}")

    random.shuffle(all_urls)
    all_urls_df = pd.DataFrame(all_urls, columns=["Values"])
    all_urls_df.to_csv(all_urls_output_file, index=False)
    print(f"All URLs saved to {all_urls_output_file}")
    """
