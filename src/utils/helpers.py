import base64
from typing import Any, Dict, Optional

import requests


def analyze_virustotal_url(url: str, api_key: str = "YOUR_VIRUSTOTAL_API_KEY") -> Optional[Dict[str, Any]]:
    """
    Analyze a URL using VirusTotal's API and retrieve the analysis results.

    Parameters:
        url (str): The URL to be analyzed.
        api_key (str): Your VirusTotal API key. Default is "YOUR_VIRUSTOTAL_API_KEY".

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the analysis results if successful,
        or None if there was an error.
    """
    VirusTotal_URL = "https://www.virustotal.com/api/v3/urls"
    # Encode the URL in base64 according to VirusTotal's requirements
    encoded_url = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    headers = {"x-apikey": api_key}

    try:
        # Make the request to VirusTotal API
        response = requests.get(
            f"{VirusTotal_URL}/{encoded_url}",
            headers=headers,
            timeout=30,
        )
        if response.status_code == 200:
            # Return the JSON response if the request is successful
            return response.json()
        else:
            print(f"Error fetching VirusTotal data for {url}, " f"response code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
