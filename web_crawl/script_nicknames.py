from multiprocessing import Pool
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import *
import os
import pathlib

url = 'https://www.mobafire.com/league-of-legends/players?sort_type=create_ts&sort_order=desc&name=&'

options = Options()
options.add_argument('--headless')
driver_path = pathlib.Path(__file__).parent / 'chromedriver.exe'


def get_users_nick(server, elo, driver):
    filt = f'server={str(server).upper()}&elo_min={str(elo).title()}&elo_max={str(elo).title()}'
    count = 1

    while 1:
        driver.get(url + filt + f'&page={count}')

        try:
            _ = driver.find_element_by_xpath('//*[@id="browse-players"]/p').text
            break
        except NoSuchElementException:
            pass

        table = driver.find_element_by_xpath('//*[@id="browse-players"]/div[2]/div/table/tbody')
        rows = table.find_elements_by_tag_name('tr')

        nicknames = []
        for element in rows:
            nicknames.append(str(element.find_elements_by_tag_name('td')[1].find_element_by_tag_name('a').text))

        file = f'{server}'
        os.makedirs(file, exist_ok=True)

        with open(f'{file}/{elo}.txt', 'a+') as writable:
            writable.write('\n'.join(nicknames))

        count += 1


def call_for_each_server(server):
    elos = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'master', 'challenger']

    driver = Chrome(str(driver_path.absolute()), chrome_options=options)
    for elo in elos:
        get_users_nick(server, elo, driver)

    driver.close()
    driver.quit()
    del driver


if __name__ == '__main__':
    servers = ['ch', 'ph', 'sea']

    with Pool(processes=13) as p:
        p.map(call_for_each_server, servers)
