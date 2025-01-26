import re
import time
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urlparse

import pandas as pd
import tldextract
import whois
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from get_data import fetch_HTML_urls
from utils.helpers import analyze_virustotal_url

stop_words = {"http", "https", "www", "com", "org", "net", "html", "php", "jsp"}

phishing_keywords = [
    "urgent",
    "verify",
    "secure",
    "immediate",
    "important",
    "confirm",
    "suspend",
    "deactivate",
    "warning",
    "alert",
    "attention",
    "critical",
    "free",
    "win",
    "prize",
    "gift",
    "claim",
    "exclusive",
    "reward",
    "jackpot",
    "congratulations",
    "lucky",
    "bonus",
    "discount",
    "promo",
    "account",
    "login",
    "password",
    "reset",
    "access",
    "registration",
    "payment",
    "invoice",
    "billing",
    "transaction",
    "refund",
    "credit_card",
    "email",
    "message",
    "inbox",
    "notification",
    "support",
    "contact_us",
    "service",
    "system_update",
    "antivirus",
    "malware",
    "scan",
    "detected",
]


def is_valid_url(url: str) -> bool:
    """
    Validate if the given string is a properly formatted URL.

    Parameters:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract features from the given URL.

    Parameters:
        url (str): The URL to extract features from.

    Returns:
        Dict[str, Any]: A dictionary containing extracted URL features.
    """
    features = {}
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(parsed_url.netloc)
    features["url_length"] = len(url)
    features["has_http"] = 1 if url.startswith(("http://", "https://")) else 0
    features["contains_ip_address"] = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed_url.netloc) else 0
    features["domain_length"] = len(domain_info.domain)
    features["subdomain_count"] = len(domain_info.subdomain.split(".")) if domain_info.subdomain else 0
    features["is_suspicious_tags"] = (
        1 if domain_info.suffix not in ["com", "org", "net", "edu", "gov", "mil", "int", "co", "us"] else 0
    )
    features["path_length"] = len(parsed_url.path)
    features["contains_special_chars"] = 1 if re.search(r'[~!@$^*()+{}<>[\]|;:"<>,]', parsed_url.path) else 0
    features["url_keyword_count"] = sum(1 for kw in phishing_keywords if kw in url.lower())
    return features


def extract_html_features(html_content: str) -> Dict[str, Any]:
    """
    Extract features from HTML content.

    Parameters:
        html_content (str): The HTML content to extract features from.

    Returns:
        Dict[str, Any]: A dictionary containing extracted HTML features.
    """
    if not html_content:
        return {"phishing_keyword_count": 0, "external_links_count": 0, "script_count": 0}
    s = BeautifulSoup(html_content, "html.parser")
    text_content = s.get_text().lower()
    keyword_count = sum(text_content.count(keyword) for keyword in phishing_keywords)
    external_links_count = sum(1 for link in s.find_all("a", href=True) if link["href"].startswith("http"))
    script_count = len(s.find_all("script"))
    return {
        "phishing_keyword_count": keyword_count,
        "external_links_count": external_links_count,
        "script_count": script_count,
    }


def extract_virustotal_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from VirusTotal API response.

    Parameters:
        data (Dict[str, Any]): VirusTotal API response data.

    Returns:
        Dict[str, Any]: A dictionary containing extracted VirusTotal features.
    """
    features = {}
    stats = data.get("attributes", {}).get("last_analysis_stats", {})
    votes = data.get("attributes", {}).get("total_votes", {})
    title = data.get("attributes", {}).get("title", "").lower()
    features["malicious_count"] = stats.get("malicious", 0)
    features["suspicious_count"] = stats.get("suspicious", 0)
    features["harmless_count"] = stats.get("harmless", 0)
    features["votes_harmless"] = votes.get("harmless", 0)
    features["votes_malicious"] = votes.get("malicious", 0)
    features["has_login_keyword"] = int("login" in title)
    features["has_secure_keyword"] = int("secure" in title)
    features["risk_score"] = (
        features["malicious_count"] * 3 + features["suspicious_count"] * 2 - features["harmless_count"]
    )
    return features


def process_whois_features(whois_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process WHOIS record to extract features.

    Parameters:
        whois_record (Dict[str, Any]): WHOIS record.

    Returns:
        Dict[str, Any]: A dictionary containing processed WHOIS features.
    """
    try:
        creation_date = whois_record.get("created")
        expiration_date = whois_record.get("expires")
        creation_date = creation_date[0] if isinstance(creation_date, list) else creation_date
        expiration_date = expiration_date[0] if isinstance(expiration_date, list) else expiration_date
        domain_age_years = (datetime.now() - creation_date).days // 365 if creation_date else 0
        days_to_expire = (expiration_date - datetime.now()).days if expiration_date else -1
        is_expiring_soon = 1 if days_to_expire > 0 and days_to_expire < 30 else 0
        return {"domain_age_years": domain_age_years, "is_expiring_soon": is_expiring_soon}
    except Exception:
        return {"domain_age_years": 0, "is_expiring_soon": 0}


def get_whois_info(domain: str) -> Dict[str, Any]:
    """
    Retrieve WHOIS information for a given domain.

    Parameters:
        domain (str): The domain name.

    Returns:
        Dict[str, Any]: A dictionary containing WHOIS information.
    """
    try:
        w = whois.whois(domain)
    except Exception:
        return {"domain": domain, "registered": 0}
    return {
        "domain": domain,
        "created": w.creation_date,
        "expires": w.expiration_date,
        "updated": w.updated_date,
        "registrar": w.registrar,
        "name_servers": w.name_servers,
        "registered": 1 if w.status else 0,
    }


def batch_analyze_urls(urls: list, max_queries: int = 10) -> list:
    """
    Analyze a batch of URLs using the VirusTotal API.

    Parameters:
        urls (list): List of URLs to analyze.
        max_queries (int): Maximum number of queries to perform. Default is 10.

    Returns:
        list: List of dictionaries containing VirusTotal analysis results.
    """
    results = []
    for i, url in enumerate(urls):
        if i >= max_queries:
            print("Reached maximum allowed queries.")
            break
        print(f"Analyzing: {url}")
        vt_data = analyze_virustotal_url(url)
        features = extract_virustotal_features(vt_data.get("data", {})) if vt_data else {}
        features["url"] = url
        results.append(features)
        time.sleep(15)
    return results


def batch_process_features(urls: list) -> pd.DataFrame:
    """
    Process features for a batch of URLs by extracting various data points.

    Parameters:
        urls (list): List of URLs to process.

    Returns:
        pd.DataFrame: DataFrame containing processed features for each URL.
    """
    features_list = []
    for url in urls:
        if not is_valid_url(url):
            print(f"Skipping invalid URL: {url}")
            continue
        try:
            url_data = extract_url_features(url)
            html_data = fetch_HTML_urls(url)
            html_features = extract_html_features(html_data)
            domain = urlparse(url).netloc
            whois_info = get_whois_info(domain)
            whois_features = process_whois_features(whois_info)
            vt_data = analyze_virustotal_url(url)
            vt_features = extract_virustotal_features(vt_data.get("data", {})) if vt_data else {}

            features = {**url_data, **html_features, **whois_features, **vt_features, "url": url}
        except Exception as error:
            features = {"url": url, "error": str(error)}

        features_list.append(features)
        time.sleep(2)

    return pd.DataFrame(features_list)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing columns with errors and null values.

    Parameters:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Cleaned data with relevant columns and non-null values.
    """
    data = data.drop(columns=["error"])
    data = data.dropna()
    data = data.loc[:, data.nunique() > 1]
    data_float = data.drop(columns=["url"])
    return data_float


def prepare_and_process_nlp_data(
    full_urls: list, openphish_urls: list, output_path: str = "nlp_data.csv"
) -> pd.DataFrame:
    """
    Prepare and process NLP data from URL lists.

    Parameters:
        full_urls (list): List of full URLs.
        openphish_urls (list): List of OpenPhish URLs.
        output_path (str): Path to save the processed data. Default is "nlp_data.csv".

    Returns:
        pd.DataFrame: DataFrame containing processed NLP data.
    """
    nlp_data = pd.DataFrame(
        {"url": full_urls + openphish_urls, "label": [1] * len(full_urls) + [0] * len(openphish_urls)}
    )

    def process_url(url: str) -> list:
        """
        Tokenize and stem the URL, removing stop words.

        Parameters:
            url (str): The URL to process.

        Returns:
            list: List of processed tokens.
        """
        tokenizer = RegexpTokenizer(r"[A-Za-z]+")
        stemmer = SnowballStemmer("english")
        tokens = tokenizer.tokenize(url)
        filtered = [word for word in tokens if word.lower() not in stop_words]
        return [stemmer.stem(word) for word in filtered]

    nlp_data["stemmer_tokens"] = nlp_data["url"].apply(process_url)
    nlp_data["text_sent"] = nlp_data["stemmer_tokens"].apply(" ".join)

    nlp_data.to_csv(output_path, index=False)
    return nlp_data


def clean_numeric_features(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Clean numeric features by handling missing values and converting data types.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list): List of columns to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned numeric features.
    """
    df.fillna(0, inplace=True)
    non_numeric_cols = df.select_dtypes(exclude=["float64", "int64"]).columns
    df_numeric = df.drop(columns=non_numeric_cols)
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df[col], errors="coerce")
    df_numeric.fillna(0, inplace=True)

    return df_numeric


if __name__ == "__main__":

    print("Don't run this code. My API can only run 500 URLs a day.")
    print("Here are 1,000 URLs. It took me three days to run them.")

    """
    all_urls_output_file = os.path.abspath("../data/processed/all_urls.csv")
    tranco_output_file = os.path.abspath("../data/raw/tranco_urls.csv")
    openphish_file = os.path.abspath("../data/raw/openphish_urls.csv")

    openphish_urls = pd.read_csv(openphish_file)
    tranco_df = pd.read_csv(tranco_output_file)
    tranco_urls = list(tranco_df["URL"])
    test_openphish_urls = list(openphish_urls["URL"])
    test_tranco_urls = tranco_urls
    test_urls = test_openphish_urls + test_tranco_urls
    test_features = batch_process_features(test_urls)
    test_features['label'] = [1] * len(test_openphish_urls) + [0] * len(test_tranco_urls)
    print(test_features)
    test_features.to_csv("data/processed/final_test_features.csv", index=False)

    full_urls = list(pd.read_csv(all_urls_output_file)["Values"])
    nlp_data = prepare_and_process_nlp_data(full_urls, openphish_urls)
    """
