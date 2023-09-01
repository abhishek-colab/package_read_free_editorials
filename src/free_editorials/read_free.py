
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError
import sys


def read(url:str) -> None:
    '''
    This function helps you read editorials without subscribing
    Args:
        str: pass the url of the editorial
    Returns:
        None: prints the article

    '''
    try:
        html = urlopen(url)      
    except HTTPError as e:
        print(e)
    except URLError as e:
        print('The server could not be found!')
    else:
        bs = BeautifulSoup(html, 'lxml')  #html5lib
        texts = bs.find_all('p')

        for text in texts:
            print(text.get_text())

try: 
    url = sys.argv[1]
    
    try:
        read(url)
    except:
        print("check url")

except:
        print("check or pass url")


