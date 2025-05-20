# please run "pip install -r requirements.txt" to install the necessary packages before running the code.

import time
import pandas as pd
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

print("Starting script...")

# ---------------web---------------------------------------------------------------
# 1. Set up the Selenium WebDriver
# ------------------------------------------------------------------------------
# 1) Create a Service object with path to your ChromeDriver
service = Service(ChromeDriverManager().install())

# 2 Pass the Service object to the WebDriver constructor
driver = webdriver.Chrome(service=service)

print("Opening website...")
driver.get("https://bu.campuslabs.com/engage/organizations")
print("Website loaded")

# ------------------------------------------------------------------------------
# 2. If there's a "Load More" button or infinite scroll, keep loading more
# ------------------------------------------------------------------------------
print("Looking for Load More button...")
load_attempts = 0
while True:
    try:
        # If the page has a 'Load More' button, locate and click it
        # This is an EXAMPLE XPATH -- adjust as needed
        load_more_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//span[contains(text(),"Load More")]'))
        )
        load_more_button.click()
        load_attempts += 1
        print(f"Clicked Load More button {load_attempts} time(s)")
        
        # Wait a bit for new items to load
        time.sleep(2)
    except Exception as e:
        print(f"No more Load More buttons found or error: {str(e)}")
        break

# ------------------------------------------------------------------------------
# 3. Extract organization data
# ------------------------------------------------------------------------------
# Example lists to store the scraped data
org_names = []
org_descriptions = []
org_links = []

# Identify all organization cards
# NOTE: This CSS selector or XPath must match the actual organization card elements.
# You need to inspect the page and find the correct selector.
print("Waiting for page to settle...")
time.sleep(2)  # short sleep to ensure all cards are rendered

print("Finding organization cards...")
organization_cards = driver.find_elements(By.CSS_SELECTOR, "a[href*='/engage/organization']")
print(f"Found {len(organization_cards)} organization cards")

for card in organization_cards:
    try:
        # Extract the organization's name and description
        # Organization cards usually have the name and description text separated by newlines
        full_text = card.text.strip()
        text_parts = full_text.split('\n', 1)  # Split at first newline to separate name and description
        
        # Get the name (first part)
        org_name = text_parts[0].strip()
        
        # Get the description (second part) if available
        org_description = text_parts[1].strip() if len(text_parts) > 1 else ""
        
        # Get the link from the organization's card (the href attribute)
        org_link = card.get_attribute("href")
        
        if org_name and org_link:
            org_names.append(org_name)
            org_descriptions.append(org_description)
            org_links.append(org_link)
            print(f"Added: {org_name} - {org_link}")
    except Exception as e:
        print(f"Error processing card: {str(e)}")

print(f"Total organizations found: {len(org_names)}")

# ------------------------------------------------------------------------------
# 4. Save the data to Excel using pandas
# ------------------------------------------------------------------------------
if org_names:
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'Organization Name': org_names,
        'URL': org_links,
        'Description': org_descriptions
    })

    # Add an index column starting from 1
    df.insert(0, 'Index', range(1, len(df) + 1))

    print("Saving to CSV...")
    df.to_csv("bu_organizations.csv", index=False)
    print("File saved")
else:
    print("No organizations found, not creating CSV file")

# Close the browser
driver.quit()

print("Scraping complete! Data saved to 'bu_organizations.csv'.")