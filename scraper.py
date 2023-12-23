import csv
import logging
import os
import re
import time
from datetime import datetime
from io import StringIO
from tempfile import mkdtemp

import boto3
import pytz
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from Bet import Bet

ENV = os.environ.get("ENV", "local")

num_errors = 0
list_of_bets = []
processed_rows = set()  # To keep track of what you've already processed
cur_date = None

URL = "https://www.winner.co.il/"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def parse_event_title(child):
    # print('Parsing event title')
    market_header = child.find_element(By.XPATH, ".//div[@class='market-header']")
    if market_header:
        # print('debug1')
        market_header_title = market_header.find_element(
            By.XPATH, ".//div[@class='market-header-title']"
        )
        if market_header_title:
            # print('debug2')
            market_header_title_content = market_header_title.find_element(
                By.XPATH, ".//div[@class='market-header-title-content']"
            )
            if market_header_title_content:
                # print('debug3')
                try:
                    market_header_title_content_details = (
                        market_header_title_content.find_element(
                            By.XPATH,
                            ".//h4[@class='market-header-title-content-details']",
                        )
                    )
                except:
                    print("error: no market_header_title_content_details")
                if market_header_title_content_details:
                    # print('debug4')
                    icon = market_header_title_content_details.find_element(
                        By.XPATH, ".//span[@class='icon']"
                    )
                    if icon:
                        # print('debug5')
                        type_element = icon.find_element(
                            By.XPATH, ".//span[@class='sr-only']"
                        )
                        type = type_element.text
                    title = market_header_title_content_details.find_element(
                        By.XPATH, ".//span[@class='title']"
                    )
                    if title:
                        # go to span which class contains the text time
                        # print('debug6')
                        time_element = title.find_element(
                            By.XPATH, ".//span[contains(@class, 'time')]"
                        )
                        # get the value of the span
                        if time_element:
                            time_text = time_element.text
                        # print('debug7')
                        market_description = title.find_element(
                            By.XPATH, ".//span[contains(@class, 'market-description')]"
                        )

                        if market_description:
                            # get the value of the span of class league
                            league_element = market_description.find_element(
                                By.XPATH, ".//span[@class='league']"
                            )
                            league = league_element.text
                            market_type_element = market_description.find_element(
                                By.XPATH, ".//span[@class='market-type']"
                            )
                            market_type = market_type_element.text
    bet_event = league + parse_hebrew(market_type).replace("1X2 - ", "")
    return type, time_text, bet_event


def parse_event_odds(child):
    # print('Parsing event odds')
    market_btns = child.find_element(By.XPATH, ".//div[@class='market-btns']")
    if market_btns:
        print('DEBUG1')
        btn_group_toggle = child.find_element(
            By.XPATH, ".//div[@class='btn-group-toggle']"
        )
        if btn_group_toggle:
            # for each label print the aria-label
            # print('DEBUG2')
            labels = btn_group_toggle.find_elements(By.XPATH, ".//label")
            options = []
            for label in labels:
                outcome = label.find_element(By.XPATH, ".//div[@class='outcome']")
                if outcome:
                    # print the span of class which contains the string hasHebrewCharacters and span of class ratio
                    hasHebrewCharacters = outcome.find_element(
                        By.XPATH, ".//span[contains(@class, 'hasHebrewCharacters')]"
                    )
                    team = hasHebrewCharacters.text
                    team = parse_hebrew(team)
                    ratio = outcome.find_element(By.XPATH, ".//span[@class='ratio']")
                    ratio_value = ratio.text
                    options.append({"team": team, "ratio": ratio_value})
        else:
            print("error: no btn_group_toggle")
    else:
        print("error: no market_btns")
    
    option1, ratio1 = options[0]["team"], options[0]["ratio"]
    option2, ratio2 = options[1]["team"], options[1]["ratio"]
    option3, ratio3 = None, None
    if len(options) > 2:
        option3, ratio3 = options[2]["team"], options[2]["ratio"]

    return option1, ratio1, option2, ratio2, option3, ratio3


def parse_sport_event(child):
    # print('Parsing sport event')
    type, time_text, bet_event = parse_event_title(child)
    option1, ratio1, option2, ratio2, option3, ratio3 = parse_event_odds(child)
    bet = Bet(
        type_convertor(type),
        cur_date,
        time_text,
        bet_event,
        option1,
        ratio1,
        option2,
        ratio2,
        option3,
        ratio3,
    )
    if str(bet) not in processed_rows and "סך הכל שערים" not in bet.event and 'סה"כ' not in bet.event:
        processed_rows.add(str(bet))
        list_of_bets.append(bet)
    print(bet.event_date, bet.event[::-1], option1[::-1], option2[::-1], option3[::-1])


def convert_to_date_format(input_str):
    current_year = datetime.now().year
    return datetime.strptime(f"{input_str}.{current_year}", "%d.%m.%Y").strftime(
        "%Y-%m-%d"
    )


def parse_date(nested_child):
    # print('Parsing date')
    # Locate the date element (assuming it's a span or div, adapt as needed)
    date_element = nested_child[0].find_element(
        By.XPATH, ".//span | .//div"
    )  # Change the tag name as per your HTML structure
    if date_element:
        global cur_date
        cur_date = convert_to_date_format(date_element.text.split("|")[1])
    else:
        print("error: Date not found")


def parse_row(row):
    # print('Parsing row')
    childs = row.find_elements(By.XPATH, ".//div")
    if not childs:
        print("error: no childs")
        return
    if len(childs) == 1:
        # print('Found add, skipping')
        return
    child = childs[0]
    nested_child = child.find_elements(By.XPATH, ".//div")
    if nested_child and "market-date" in nested_child[0].get_attribute("class"):
        parse_date(nested_child)
    if child.get_attribute("class") == "market  market-01":
        parse_sport_event(child)
#//*[@id="WinnerLine"]/div[2]/div[3]/div/div/div[1]/div/div/div[3]/div/div[1]/div[1]/div/h4/span[2]/span[3]/span[1]
#//*[@id="WinnerLine"]/div[2]/div[3]/div/div/div[1]/div/div/div[3]
#//*[@id="WinnerLine"]/div[2]/div[3]/div/div/div[1]/div/div/div[4]

def parse_sport(driver, temp, i):
    # print('Parsing sport')
    scrollable_element_xpath = (
        '//*[@id="WinnerLine"]/div[2]/div[3]/div/div/div[1]/div/div'
    )
    scrollable_element = driver.find_element(By.XPATH, scrollable_element_xpath)
    scrollable_element.click()
    time.sleep(0.1)
    main_element = driver.find_element(
        By.XPATH, '//*[@id="WinnerLine"]/div[2]/div[3]/div/div/div[1]/div/div'
    )
    action = ActionChains(driver)
    for i in range(22 if i==2 else 8):
        child_elements = main_element.find_elements(By.XPATH, ".//div[@role='row']")
        for row in child_elements:
            try:
                parse_row(row)
                pass
            except Exception as e:
                print(
                    f"\n\n\n\n\nAn exception occurred while processing a sport event: {e}\n\n\n\n\n\n"
                )
                global num_errors
                num_errors += 1

        # print('scrolling down -----------------------------------')
        for i in range(2):
            action.send_keys(Keys.PAGE_DOWN).perform()
            time.sleep(0.05)

        # time.sleep(2)


def type_convertor(text):
    if "כדורגל," == text:
        return "Soccer"
    elif "כדורסל," == text:
        return "Basketball"
    elif "טניס," == text:
        return "Tennis"
    elif "כדוריד," == text:
        return "Handball"
    elif "כדורגל אמריקאי," == text:
        return "American Football"
    elif "בייסבול," == text:
        return "Baseball"
    else:
        return "Other"


def parse_hebrew(text):
    # Remove Unicode control characters
    text = re.sub(r"[\u202A\u202C\u202E]", "", text)
    # Switch opening and closing parentheses
    # text = text.replace("(", "TEMP_PLACEHOLDER")
    # text = text.replace(")", "(")
    # text = text.replace("TEMP_PLACEHOLDER", ")")
    return text


def write_bets_to_s3(bets: list[dict]):
    # Get the current time in GMT+3
    gmt3 = pytz.timezone("Etc/GMT-3")  # Note the negative sign, it's an offset from GMT
    current_time_gmt3 = datetime.now(gmt3)
    cur_date = current_time_gmt3.strftime("%Y-%m-%d")
    cur_time = current_time_gmt3.strftime("%H:%M:%S")
    run_time = f"{cur_date} {cur_time}"

    for bet in bets:
        bet["run_time"] = run_time

    csv_buffer = StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=bets[0].keys())
    csv_writer.writeheader()
    for row in bets:
        csv_writer.writerow(row)

    # Write a column run_time with the current time
    csv_buffer.seek(0)
    csv_content = csv_buffer.getvalue()

    if ENV == "prod":
        # Upload to S3
        s3 = boto3.resource("s3")
        bucket_name = "boaz-winner-prod"
        file_name = f"odds/date_parsed={cur_date}/{cur_time}.csv"
        object = s3.Object(bucket_name, file_name)
        object.put(Body=csv_content)
    else:
        # save to csv
        with open(f"scraper_output.csv", "w") as f:
            f.write(csv_content)


def configure_chrome_options():
    options = webdriver.ChromeOptions()
    if ENV == "prod":
        options.binary_location = "/opt/chrome/chrome"
        options.add_argument("--disable-gpu")
        options.add_argument("--single-process")
        options.add_argument("--no-zygote")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--window-size=480,1000")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    return options


def get_chrome_service():
    return webdriver.ChromeService("/opt/chromedriver") if ENV == "prod" else None



def process_buttons(driver, temp):
    for i in range(2, 5):  # Adjust range if needed
        try:
            print("Trying button ", i)
            button = driver.find_element(
                By.XPATH, f'//*[@id="sport-items"]/div[{i}]/label/button'
            )
            button.click()
            time.sleep(1)
            if i == 2:
                driver.get(
                    "https://www.winner.co.il/%D7%9E%D7%A9%D7%97%D7%A7%D7%99%D7%9D/%D7%95%D7%95%D7%99%D7%A0%D7%A8-%D7%9C%D7%99%D7%99%D7%9F/%D7%9B%D7%93%D7%95%D7%A8%D7%92%D7%9C/%D7%90%D7%99%D7%98%D7%9C%D7%99%D7%94,%D7%90%D7%A0%D7%92%D7%9C%D7%99%D7%94,%D7%92%D7%A8%D7%9E%D7%A0%D7%99%D7%94,%D7%99%D7%A9%D7%A8%D7%90%D7%9C,%D7%9E%D7%95%D7%A2%D7%93%D7%95%D7%A0%D7%99%D7%9D%20%D7%91%D7%99%D7%A0%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%99%D7%9D,%D7%A1%D7%A4%D7%A8%D7%93,%D7%A6%D7%A8%D7%A4%D7%AA/4336287;4336293;4336297;4336310;4336321;%D7%99%D7%A9%D7%A8%D7%90%D7%9C$%D7%9C%D7%99%D7%92%D7%AA%20%D7%94%D7%A2%D7%9C;%D7%9E%D7%95%D7%A2%D7%93%D7%95%D7%A0%D7%99%D7%9D%20%D7%91%D7%99%D7%A0%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%99%D7%9D$%D7%9C%D7%99%D7%92%D7%AA%20%D7%94%D7%90%D7%9C%D7%95%D7%A4%D7%95%D7%AA;%D7%9E%D7%95%D7%A2%D7%93%D7%95%D7%A0%D7%99%D7%9D%20%D7%91%D7%99%D7%A0%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%99%D7%9D$%D7%9C%D7%99%D7%92%D7%AA%20%D7%94%D7%90%D7%9C%D7%95%D7%A4%D7%95%D7%AA%20%D7%94%D7%90%D7%A1%D7%99%D7%90%D7%AA%D7%99%D7%AA;%D7%9E%D7%95%D7%A2%D7%93%D7%95%D7%A0%D7%99%D7%9D%20%D7%91%D7%99%D7%A0%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%99%D7%9D$%D7%9C%D7%99%D7%92%D7%AA%20%D7%94%D7%90%D7%9C%D7%95%D7%A4%D7%95%D7%AA%20%D7%94%D7%90%D7%A4%D7%A8%D7%99%D7%A7%D7%90%D7%99%D7%AA;%D7%9E%D7%95%D7%A2%D7%93%D7%95%D7%A0%D7%99%D7%9D%20%D7%91%D7%99%D7%A0%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%99%D7%9D$%D7%A7%D7%95%D7%A0%D7%A4%D7%A8%D7%A0%D7%A1%20%D7%9C%D7%99%D7%92"
                )
                time.sleep(3) 
            parse_sport(driver, temp, i)
            global num_errors
            # print("FOR TIME: ", temp, "NUM ERRORS: ", num_errors)
            num_errors = 0
        except Exception as e:
            print(f"An exception occurred while processing button {i}: {e}")


def bet_to_dict(bet):
    bet_dict = {
        "Type": bet.type,
        "Date": bet.event_date,
        "Time": bet.time,
        "Event": bet.event,
        "Option1": bet.option1,
        "Ratio1": bet.ratio1,
        "Option2": bet.option2,
        "Ratio2": bet.ratio2,
    }
    if bet.option3 and bet.ratio3:
        bet_dict["Option3"] = bet.option3
        bet_dict["Ratio3"] = bet.ratio3
    else:
        bet_dict["Option3"] = None
        bet_dict["Ratio3"] = None
    return bet_dict

def scrape_winner_with_selenium():
    options = configure_chrome_options()
    service = get_chrome_service()
    with webdriver.Chrome(options=options, service=service) as driver:
        driver.get(URL)
        driver.implicitly_wait(10)
        time.sleep(7)
        process_buttons(driver, temp=0.1)

        bet_dicts = [bet_to_dict(bet) for bet in list_of_bets]
        num_rows = len(list_of_bets)
        write_bets_to_s3(bet_dicts)
        return num_rows
    
def main(event, context):
    print("enviroment: ", ENV)
    num_rows = scrape_winner_with_selenium()
    response = {
        "statusCode": 200,
        "body": "Successfully scraped " + str(num_rows) + " rows.",
    }
    logger.info("NUM ERRORS:" + str(num_errors))
    print("NUM ROWS: ", num_rows)
    return response


if __name__ == "__main__":
    main(None, None)
