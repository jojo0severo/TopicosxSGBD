from multiprocessing import Pool
from selenium.webdriver import Chrome
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import *
import os
import inflect
import pathlib
import random

url = 'https://teemo.gg/player'

options = Options()
options.add_argument('--headless')
driver_path = pathlib.Path(__file__).parent / 'chromedriver.exe'


def get_mastery_champions(server, player, driver):
    driver.get(f'{url}/champion-masteries/{server}/{player}')
    sort = '//*[@id="champion-mastery-sorter"]'
    dropdown = Select(driver.find_element_by_xpath(sort))
    dropdown.select_by_visible_text('Mastery Points')

    champions = []
    for i in range(1, 4):
        xpath = f'//*[@id="champ-mastery-items"]/div/div[2]/div/div[{i}]/p'
        if try_to_find_element(driver, xpath, 'xpath'):
            champions.append(driver.find_element_by_xpath(xpath).text)
        else:
            exit('Deu erro na mastery')

    return champions


def get_player_status(server, player, driver):
    driver.get(f'{url}/statistics/{server}/{player}')
    score_board = driver.find_element_by_id('tm-scoreboard')
    if try_to_find_element(driver, 'tm-scoreboard', 'id'):
        stats = score_board.find_elements_by_class_name('points')
        return [
            stats[0].text,
            stats[1].text,
            stats[2].text,
            stats[3].text,
            stats[4].text,
            stats[5].text,
            stats[6].text,
            stats[7].text,
            stats[8].text,
            stats[9].text,
            stats[10].text
        ]
    else:
        exit('Deu erro nos stats')


def get_player_elo(server, player, driver, elo):
    driver.get(f'{url}/resume/{server}/{player}')
    xpath = '/html/body/div/section[1]/div/div[1]/div[2]/div[2]/p[1]/b'
    if try_to_find_element(driver, xpath, 'xpath'):
        current = driver.find_element_by_xpath(xpath).text
        return elo if current == 'Unranked' else current

    else:
        raise IndexError


def iterate_through_servers(path):
    elos = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'master', 'challenger']

    if path.find('/') != '-1':
        server = path.split('\\')[-1]
    else:
        server = path.split('/')[-1]

    driver = Chrome(executable_path=str(driver_path.absolute()), options=options)
    resp = []
    for elo in elos:
        try:
            with open(f'{path}/{elo}.txt', 'r') as players:
                for player in players.read().split('\n'):
                    try:
                        actual_elo = get_player_elo(server, player, driver, elo)
                    except IndexError:
                        continue

                    status = get_player_status(server, player, driver)
                    champions = get_mastery_champions(server, player, driver)

                    resp.append((actual_elo, * status, *champions))
        except FileNotFoundError:
            pass

    path = pathlib.Path(path).parent / f'players_{server}'
    if not os.path.exists(str(path.absolute())):
        os.mkdir(str(path.absolute()))
        open(f'players_{server}/file.txt', 'a').close()

    writable_players = []
    with open(f'players_{server}/file.txt', 'r') as file:
        lines = file.read().split('\n')
        writable_players.extend(lines)
        for i in resp:
            if str(i) not in lines:
                writable_players.append(str(i))

    with open(f'players_{server}/file.txt', 'w') as file:
        file.write('\n'.join(writable_players))

    driver.close()
    driver.quit()


def try_to_find_element(driver, identificador, kind):
    import time
    start = time.time()

    identificador = identificador.replace('"', "'")
    while time.time() - start < 5:
        try:
            _ = exec(f'driver.find_element_by_{kind}("{identificador}").text')
            return True
        except NoSuchElementException:
            pass

    return False


def get_folders(path):
    dirs = []
    for directory in os.listdir(str(path.absolute())):
        if directory != 'venv' and not directory.endswith('exe') and not directory.endswith(
                'py') and directory != '.idea' and not directory.startswith('player'):
            dirs.append(str((path / directory).absolute()))

    return dirs


def split_servers():
    folders = random.shuffle(get_folders(pathlib.Path()))

    with Pool(processes=12) as p:
        p.map(func=iterate_through_servers, iterable=folders)


if __name__ == '__main__':
    eng = inflect.engine()
    for i in range(1, 11):
        print(f'Started {eng.number_to_words(eng.ordinal(i))} collection, trying again...')
        split_servers()
