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

num_errors = 0
list_of_bets = []
processed_rows = set()  # To keep track of what you've already processed
cur_date = None

URL = "https://www.winner.co.il/"
driver_path = './chromedriver_mac64/chromedriver'  # for Linux and macOS
#logging.basicConfig(level=logging.DEBUG)

def parse_event_title(child):
    #print('Parsing event title')
    market_header = child.find_element(By.XPATH, ".//div[@class='market-header']")
    if market_header:
        # print('debug1')
        market_header_title = market_header.find_element(By.XPATH, ".//div[@class='market-header-title']")
        if market_header_title:
            # print('debug2')
            market_header_title_content = market_header_title.find_element(By.XPATH, ".//div[@class='market-header-title-content']")
            if market_header_title_content:
                # print('debug3')
                try:
                    market_header_title_content_details = market_header_title_content.find_element(By.XPATH, ".//h4[@class='market-header-title-content-details']")
                except:
                    print('error: no market_header_title_content_details')
                if market_header_title_content_details:
                    # print('debug4')
                    icon = market_header_title_content_details.find_element(By.XPATH, ".//span[@class='icon']")
                    if icon:
                        #print('debug5')
                        type_element = icon.find_element(By.XPATH, ".//span[@class='sr-only']")
                        type=type_element.text
                    title = market_header_title_content_details.find_element(By.XPATH, ".//span[@class='title']")
                    if title:
                        #go to span which class contains the text time
                        #print('debug6')
                        time_element = title.find_element(By.XPATH, ".//span[contains(@class, 'time')]")
                        #get the value of the span
                        if time_element:
                            time_text = time_element.text
                        #print('debug7')
                        market_description = title.find_element(By.XPATH, ".//span[contains(@class, 'market-description')]")
                        
                        if market_description:
                            #get the value of the span of class league
                            league_element = market_description.find_element(By.XPATH, ".//span[@class='league']")
                            league=league_element.text
                            market_type_element = market_description.find_element(By.XPATH, ".//span[@class='market-type']")
                            market_type=market_type_element.text
    bet_event = parse_hebrew(market_type[::-1])+ parse_hebrew((league)[::-1])
    return type, time_text, bet_event

def parse_event_odds(child):
    #print('Parsing event odds')
    market_btns = child.find_element(By.XPATH, ".//div[@class='market-btns']")
    if market_btns:
        #print('DEBUG1')
        btn_group_toggle=child.find_element(By.XPATH, ".//div[@class='btn-group-toggle']")
        if btn_group_toggle:
            #for each label print the aria-label
            #print('DEBUG2')
            labels = btn_group_toggle.find_elements(By.XPATH, ".//label")
            options=[]
            for label in labels:
                outcome = label.find_element(By.XPATH, ".//span[@class='outcome']")
                if outcome:
                    #print the span of class which contains the string hasHebrewCharacters and span of class ratio
                    hasHebrewCharacters = outcome.find_element(By.XPATH, ".//span[contains(@class, 'hasHebrewCharacters')]")
                    team = hasHebrewCharacters.text
                    team = parse_hebrew(team)
                    ratio = outcome.find_element(By.XPATH, ".//span[@class='ratio']")
                    ratio_value = ratio.text
                    options.append({'team': team, 'ratio': ratio_value})
    #print('DEBUG3')
    for option in options:
        if 'ץייווש' in option['team']:
            print('found')
    option1, ratio1 = options[0]['team'], options[0]['ratio']
    option2, ratio2 = options[1]['team'], options[1]['ratio']
    option3, ratio3 = None, None
    if len(options) > 2:
        option3, ratio3 = options[2]['team'], options[2]['ratio']
    return option1, ratio1, option2, ratio2, option3, ratio3

def parse_sport_event(child):
    #print('Parsing sport event')
    type, time_text, bet_event = parse_event_title(child)
    option1, ratio1, option2, ratio2, option3, ratio3 = parse_event_odds(child)
    
    bet = Bet(type_convertor(type), cur_date, time_text, bet_event, option1, ratio1, option2, ratio2, option3, ratio3)
    if str(bet) not in processed_rows:
        processed_rows.add(str(bet))
        list_of_bets.append(bet)
    #print(bet.event_date, bet.event, option1, option2, option3)


def parse_date(nested_child):
    #print('Parsing date')
    # Locate the date element (assuming it's a span or div, adapt as needed)
    date_element = nested_child[0].find_element(By.XPATH, ".//span | .//div")  # Change the tag name as per your HTML structure
    if date_element:
        global cur_date
        cur_date = date_element.text.split('|')[1]
    else:
        print('error: Date not found')


def parse_row(row):
    #print('Parsing row')
    childs = row.find_elements(By.XPATH, ".//div")
    if not childs:
        print('error: no childs')
        return
    if len(childs) == 1:
        #print('Found add, skipping')
        return
    child = childs[0]
    nested_child = child.find_elements(By.XPATH, ".//div")
    if nested_child and 'market-date' in nested_child[0].get_attribute('class'):
        parse_date(nested_child)
    if child.get_attribute('class') == 'market  market-01':
        parse_sport_event(child)


def parse_sport(driver, temp):
    #print('Parsing sport')
    scrollable_element_xpath = '//*[@id="WinnerLine"]/div[2]/div[2]/div/div/div[1]/div/div'
    scrollable_element = driver.find_element(By.XPATH,scrollable_element_xpath)
    scrollable_element.click()

    time.sleep(1)
    main_element = driver.find_element(By.XPATH, '//*[@id="WinnerLine"]/div[2]/div[2]/div/div/div[1]/div/div')        
    action = ActionChains(driver)
    
    for i in range(25):
        child_elements = main_element.find_elements(By.XPATH, ".//div[@role='row']")
        for row in child_elements:
            try:
                parse_row(row)
                pass
            except Exception as e:
                print(f"\n\n\n\n\nAn exception occurred while processing a sport event: {e}\n\n\n\n\n\n")
                global num_errors
                num_errors += 1

        #print('scrolling down -----------------------------------')
        for i in range(2):
            action.send_keys(Keys.PAGE_DOWN).perform()
            time.sleep(0.2)
            
        #time.sleep(2)


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

    # Initialize the Chrome WebDriver and open the website
    with    webdriver.Chrome(options=chrome_options) as driver:
        driver.get(URL)
        # Waiting for the content to load. Adjust the waiting time accordingly.
        driver.implicitly_wait(22)
        time.sleep(5)

        for i in range(2, 3):  # Looping from 2 to 7
            try:
                for temp in [2]:
                    print('Trying button ', i)
                    button = driver.find_element(By.XPATH, f'//*[@id="sport-items"]/div[{i}]/label/button')
                    button.click()
                    time.sleep(2)
                    parse_sport(driver, temp)
                    global num_errors
                    print("FOR TIME: ", temp, "NUM ERRORS: ", num_errors)
                    num_errors = 0
            except Exception as e:
                print(f"An exception occurred while processing button {i}: {e}")

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
    print('NUM ERRORS: ', num_errors)
