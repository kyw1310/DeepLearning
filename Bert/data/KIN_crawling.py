import numpy as np
import pandas as pd
from selenium import webdriver as wd
import time
from bs4 import BeautifulSoup as bs
import requests
import urllib
import re

default_url = "https://kin.naver.com"

data = []
page = 1

s, e = 0, 0
while True:
    try:
        if page % 10 == 0:
            print("Success:", s, "Error:", e)
        
        url = 'https://kin.naver.com/qna/list.nhn?dirId=40107&queryTime=2021-06-14%2013%3A37%3A35&page={}'.format(page)
        req = requests.get(url)
        html = req.text
        soup = bs(html, 'html.parser')
        question_items = soup.find_all("a", {"rel" : "KIN"})
        for item in question_items:
            question_url = default_url + item['href']
            q_req = requests.get(question_url)
            q_text = q_req.text
            q_html = bs(q_text, 'html.parser')
            head = q_html.find("div", {"class" : "title"}).text.replace('\t', '').replace('\n', '')
            body = q_html.find("div", {"class" : "c-heading__content"}).text.replace('\t', '').replace('\n', '')
            data.append([head, body])
            
        time.sleep(1)
        page += 1
        s += 1
    except:
        e += 1
        continue