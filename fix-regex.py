import re
import pickle


def convert_url_pattern_to_regex(url_pattern):
    """
    Convert a URL pattern like 'https://*.example.com' to a regex
    that matches with and without subdomains
    """
    url_pattern = url_pattern.strip("'\"")
    
    if url_pattern.startswith('https://*.'):
        domain = url_pattern[10:]  # Remove 'https://*.'
    elif url_pattern.startswith('http://*.'):
        domain = url_pattern[9:]   # Remove 'http://*.'
    else:
        # Handle cases without wildcard
        domain = url_pattern.replace('https://', '').replace('http://', '')
    escaped_domain = re.escape(domain)
    
    # - http or https
    # - optional subdomains
    # - the main domain
    # - optional path
    regex_pattern = f"^https?://([^./]+\\.)*{escaped_domain}(/.*)?$"
    
    return regex_pattern

with open(r"C:\Users\arkma\Sqrt-1\putting the pro in programming\0. Ongoing Projects\claim-extraction\data\reliable-sources.pkl", "rb") as f:
    reliable_sources = pickle.load(f)

regexes = [convert_url_pattern_to_regex(url) for url in reliable_sources]
print(regexes)
combined_pattern = re.compile('|'.join(regexes))
print(combined_pattern)

with open(r"C:\Users\arkma\Sqrt-1\putting the pro in programming\0. Ongoing Projects\claim-extraction\data\compiled_regex.pkl", "wb") as f:
    pickle.dump(combined_pattern, f)