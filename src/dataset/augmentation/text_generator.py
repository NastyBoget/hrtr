import random
import re
from typing import List

import requests
from bs4 import BeautifulSoup


class TextGenerator:

    def __init__(self, max_words_number: int = 3) -> None:
        self.not_allowed_symbols = re.compile(
            r'[^АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдежзийклмнопрстуфхцчшщьыъэюя0123456789.!"%(),\-?:; ]')

        self.title_url = "https://ru.wikipedia.org/w/api.php?origin=*&action=query&format=json&list=random&rnlimit=1&rnnamespace=0"
        self.article_url = "https://ru.wikipedia.org/w/api.php?origin=*&action=parse&format=json&page={title}&prop=text"

        self.max_words_number = max_words_number
        self.texts = []

    def get_text(self) -> str:
        if len(self.texts) > 0:
            return self.texts.pop()

        while len(self.texts) == 0:
            self.__fill_texts()

        return self.texts.pop()

    def __fill_texts(self) -> None:
        words = []
        while len(words) == 0:
            try:
                words = self.__get_words()
            except Exception:
                words = []

        while len(words) > self.max_words_number:
            sentence_len = random.randint(1, self.max_words_number)
            self.texts.append(" ".join(words[:sentence_len]))
            words = words[sentence_len:]

    def __get_words(self) -> List[str]:
        # 1 - Get random title of the article in Wikipedia
        title_result = requests.post(self.title_url)
        title_result_dict = title_result.json()
        title = title_result_dict["query"]["random"][0]["title"]

        # 2 - Get text the article
        article_result = requests.post(self.article_url.format(title=title))
        article_result_dict = article_result.json()
        article = article_result_dict["parse"]["text"]['*']
        bs = BeautifulSoup(article, 'html.parser')
        article_text = bs.get_text()

        # 3 - Clear text of the article from unused symbols
        article_text_fixed = re.sub(r"[«»]", '"', article_text)
        article_text_fixed = re.sub(self.not_allowed_symbols, " ", article_text_fixed)
        article_text_fixed = re.sub(r"\s[.!\"%(),\-?:;]\s", " ", article_text_fixed)
        article_text_fixed = re.sub(r"\s+", " ", article_text_fixed)

        return list(set(article_text_fixed.split(" ")))
