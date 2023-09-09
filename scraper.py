from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
import re
import pandas as pd
from Bet import Bet
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import logging

URL = "https://www.winner.co.il/"
driver_path = './chromedriver_mac64/chromedriver'  # for Linux and macOS
#logging.basicConfig(level=logging.DEBUG)



def type_convertor(text):
    if 'כדורגל,' == text:
        return 'Soccer'
    elif 'כדורסל,' == text:
        return 'Basketball'
    elif 'טניס,' == text:
        return 'Tennis'
    elif 'כדוריד,' == text:
        return 'Handball'
    elif 'כדורגל אמריקאי,' == text:
        return 'American Football'
    elif 'בייסבול,' == text:
        return 'Baseball'
    else:
        return 'Other'
    
def parse_hebrew(text):
     # Remove Unicode control characters
    text = re.sub(r'[\u202A\u202C\u202E]', '', text)
    # Switch opening and closing parentheses
    text = text.replace("(", "TEMP_PLACEHOLDER")
    text = text.replace(")", "(")
    text = text.replace("TEMP_PLACEHOLDER", ")")
    return text

def scrape_winner_with_selenium():
    # Configuring options for Chrome in headless mode (no GUI)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")

    list_of_bets = []
    processed_rows = set()  # To keep track of what you've already processed

    # Initialize the Chrome WebDriver and open the website
    with    webdriver.Chrome(options=chrome_options) as driver:
        driver.get(URL)
        # Waiting for the content to load. Adjust the waiting time accordingly.
        driver.implicitly_wait(22)
        time.sleep(5)

        for i in range(2, 4):  # Looping from 2 to 7
            try:
                print('Trying button ', i)
                #click the button at xpath //*[@id="sport-items"]/div[2]/label/button
                button = driver.find_element(By.XPATH, f'//*[@id="sport-items"]/div[{i}]/label/button')
                button.click()

                time.sleep(2)

                scrollable_element_xpath = '//*[@id="WinnerLine"]/div[2]/div[2]/div/div/div[1]/div/div'
                scrollable_element = driver.find_element(By.XPATH,scrollable_element_xpath)
                scrollable_element.click()

                time.sleep(1)
                cur_date = None
                main_element = driver.find_element(By.XPATH, '//*[@id="WinnerLine"]/div[2]/div[2]/div/div/div[1]/div/div')        
                action = ActionChains(driver)
                
                for i in range(25):
                    child_elements = main_element.find_elements(By.XPATH, ".//div[@role='row']")
                    for row in child_elements:
                        #go to the child div
                        childs = row.find_elements(By.XPATH, ".//div")
                        if not childs:
                            print('error: no childs')
                            continue
                        if len(childs) == 1:
                            print('Found add, skipping')
                            continue
                        child = childs[0]
                        nested_child = child.find_elements(By.XPATH, ".//div")
                        if nested_child and 'market-date' in nested_child[0].get_attribute('class'):
                            # Locate the date element (assuming it's a span or div, adapt as needed)
                            date_element = nested_child[0].find_element(By.XPATH, ".//span | .//div")  # Change the tag name as per your HTML structure
                            if date_element:
                                cur_date = date_element.text.split('|')[1]
                                if cur_date==None:
                                    cur_date = date_element.text
                                    print('Date found: ', cur_date)
                            else:
                                print('error: Date not found')
                        #if the child is div of class market  market-01
                        if child.get_attribute('class') == 'market  market-01':
                            market_header = child.find_element(By.XPATH, ".//div[@class='market-header']")
                            if market_header:
                                #go to div of class market-header-title
                                market_header_title = market_header.find_element(By.XPATH, ".//div[@class='market-header-title']")
                                if market_header_title:
                                    #go to div of class market-header-title-content
                                    market_header_title_content = market_header_title.find_element(By.XPATH, ".//div[@class='market-header-title-content']")
                                    if market_header_title_content:
                                        #go to h4 of class market-header-title-content-details
                                        market_header_title_content_details = market_header_title_content.find_element(By.XPATH, ".//h4[@class='market-header-title-content-details']")
                                        if market_header_title_content_details:
                                            #go to span of class icon
                                            icon = market_header_title_content_details.find_element(By.XPATH, ".//span[@class='icon']")
                                            if icon:
                                                #get the value of span of class sr-only
                                                type_element = icon.find_element(By.XPATH, ".//span[@class='sr-only']")
                                                type=type_element.text
                                            title = market_header_title_content_details.find_element(By.XPATH, ".//span[@class='title']")
                                            if title:
                                                #go to span which class contains the text time
                                                time_element = title.find_element(By.XPATH, ".//span[contains(@class, 'time')]")
                                                #get the value of the span
                                                if time_element:
                                                    time_text = time_element.text
                                                #go to span of class market-description 
                                                market_description = title.find_element(By.XPATH, ".//span[contains(@class, 'market-description')]")
                                                if market_description:
                                                    #get the value of the span of class league
                                                    league_element = market_description.find_element(By.XPATH, ".//span[@class='league']")
                                                    league=league_element.text
                                                    #get the value of the span of class market-type
                                                    market_type_element = market_description.find_element(By.XPATH, ".//span[@class='market-type']")
                                                    market_type=market_type_element.text

                            #go to the child of class market-btns
                            market_btns = child.find_element(By.XPATH, ".//div[@class='market-btns']")
                            if market_btns:
                                btn_group_toggle=child.find_element(By.XPATH, ".//div[@class='btn-group-toggle']")
                                if btn_group_toggle:
                                    #for each label print the aria-label
                                    labels = btn_group_toggle.find_elements(By.XPATH, ".//label")
                                    options=[]
                                    for label in labels:
                                        #go to span of class outcome
                                        outcome = label.find_element(By.XPATH, ".//span[@class='outcome']")
                                        if outcome:
                                            #print the span of class which contains the string hasHebrewCharacters and span of class ratio
                                            hasHebrewCharacters = outcome.find_element(By.XPATH, ".//span[contains(@class, 'hasHebrewCharacters')]")
                                            team = hasHebrewCharacters.text
                                            team = parse_hebrew(team)
                                            ratio = outcome.find_element(By.XPATH, ".//span[@class='ratio']")
                                            ratio_value = ratio.text
                                            options.append({'team': team, 'ratio': ratio_value})
                            for option in options:
                                print('team: ', option['team'][::-1])
                                if 'ץייווש' in option['team']:
                                    print('found')
                            bet_event = parse_hebrew(market_type[::-1])+ parse_hebrew((league)[::-1])
                            option1, ratio1 = options[0]['team'], options[0]['ratio']
                            option2, ratio2 = options[1]['team'], options[1]['ratio']
                            option3, ratio3 = None, None
                            if len(options) > 2:
                                option3, ratio3 = options[2]['team'], options[2]['ratio']
                            bet = Bet(type_convertor(type), cur_date, time_text, bet_event, option1, ratio1, option2, ratio2, option3, ratio3)
                            if str(bet) not in processed_rows:
                                processed_rows.add(str(bet))
                                list_of_bets.append(bet)
                    print('scrolling down -----------------------------------')
                    try:
                        for i in range(2):
                            action.send_keys(Keys.PAGE_DOWN).perform()
                            time.sleep(0.1)
                    except:
                        print('error scrolling down')
                    time.sleep(5)

            except Exception as e:
                print(f"An exception occurred while processing button {i}: {e}")
                continue  # Skip to the next iteration if an exception occurs

        # Create an empty list to store the dictionaries
        bet_dicts = []

        # Convert each Bet object to a dictionary and add it to the list
        for bet in list_of_bets:
            bet_dict = {
                'Type': bet.bet_type,
                'Date': bet.event_date,
                'Time': bet.time,
                'Event': bet.event,
                'Option1': bet.option1,
                'Ratio1': bet.ratio1,
                'Option2': bet.option2,
                'Ratio2': bet.ratio2,
            }
            if bet.option3 and bet.ratio3:
                bet_dict['Option3'] = bet.option3
                bet_dict['Ratio3'] = bet.ratio3
            bet_dicts.append(bet_dict)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(bet_dicts)

        # Save the DataFrame to a CSV file
        df.to_csv('bets.csv', index=False)



if __name__ == "__main__":
    scrape_winner_with_selenium()
