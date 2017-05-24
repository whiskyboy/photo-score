# coding: utf-8

import urllib2
from bs4 import BeautifulSoup
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import logging
import time

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')

request_root_url = "http://www.sybj.com"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30"
request_headers = {'User-Agent': user_agent}
sleep_second = 1

def get_html(page_url):
    url = request_root_url + page_url
    request = urllib2.Request(url, headers=request_headers)
    response = urllib2.urlopen(request, timeout=60)
    html = response.read()
    return html

def save_html(html, filename):
    fhtml = open(filename, 'w')
    fhtml.write(html)
    fhtml.close()
    
def get_img_url(soup):
    img_tag = soup.find("img", id="imgSybj")
    if img_tag is None:
        return None
    else:
        return img_tag.get("src")
    
def get_zan_num(soup):
    zan_num_tag = soup.find("span", id="zan-num")
    if zan_num_tag is None:
        return None
    else:
        return zan_num_tag.string
    
def get_cai_num(soup):
    cai_num_tag = soup.find("span", id="cai-num")
    if cai_num_tag is None:
        return None
    else:
        return cai_num_tag.string
    
def get_prev_page(soup):
    prev_page_tag = soup.find("a", id="prepage")
    if prev_page_tag is None:
        return None
    else:
        return prev_page_tag.get("href")
    
def save_image(src_url, trg_name):
    urllib.urlretrieve(src_url, trg_name)
    
def get_img_id(url):
    _idx = url.find("&id=") + 4
    img_id = url[_idx:]
    return img_id


if __name__ == "__main__":
    #first_page = "/may.php?c=w&a=organizationCommunity&t=1&hid=1126&id=405"
    first_page = "/may.php?c=w&a=organizationCommunity&t=1&hid=1126&id=2216"
    href = first_page
    
    while href is not None:
        img_id = get_img_id(href)
        if (not href.startswith("/may.php?")) or (img_id == ""):
            break
        
        try:
            logging.info("process url=%s%s"%(request_root_url, href))
            
            html = get_html(href)
            soup = BeautifulSoup(html, 'html.parser')
            save_html(html, "./data/cache/%s.html"%img_id)
            logging.info("[html]./data/cache/%s.html has been saved."%img_id)
                
            img_url = get_img_url(soup)
            zan_num = get_zan_num(soup)
            cai_num = get_cai_num(soup)
            img_filename = "./data/img/%s_%s_%s.jpg" % (img_id, zan_num, cai_num)
            save_image(img_url, img_filename)
            logging.info("[image]%s has been saved."%img_filename)
        except Exception, e:
            logging.error(str(e))
            time.sleep(sleep_second*10)
            continue
        
        href = get_prev_page(soup)
        time.sleep(sleep_second)
    
    logging.info("Finished.")

