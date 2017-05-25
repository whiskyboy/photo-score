# coding: utf-8

import os
import urllib2
from bs4 import BeautifulSoup
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import logging
import time
import re

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')


def all_strip(s):
    return "".join(s.split())


class WebParser(object):
    def __init__(self, img_id, wait_second=1, max_retry_time=10):
        self.html_cache_path = "./data/cache/"
        self.img_path = "./data/img/"
        self.img_id = img_id
        self.soup = None
        self.request_num = 0
        self.cached_img_num = 0
        self.wait_second = wait_second
        self.max_retry_time = max_retry_time

        if -1 == self.load_html():
            logging.fatal("loading html for img_id=[%s] failed!"%self.img_id)
            self.is_loaded = False
        else:
            self.is_loaded = True
    
    def build_request_url(self):
        return "http://www.sybj.com/may.php?c=w&a=organizationCommunity&t=1&hid=1126&id=%s"%self.img_id
    
    def build_request_headers(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30"
        request_headers = {'User-Agent': user_agent}
        return request_headers

    def load_html(self):
        cache_file = self.html_cache_path + "%s.html"%self.img_id
        if os.path.exists(cache_file):
            logging.info("try load html from file=[%s]"%cache_file)
            html = open(cache_file, 'r').read()
        else:
            url = self.build_request_url()
            headers = self.build_request_headers()
            request = urllib2.Request(url, headers=headers)
            is_opened = False
            for _ in range(self.max_retry_time):
                try:
                    logging.info("try load html from url=[%s]"%url)
                    response = urllib2.urlopen(request, timeout=60)
                except Exception, e:
                    logging.warning("img_id=[%s] load html failed."%self.img_id)
                    logging.error(str(e))
                    time.sleep(self.wait_second * 10)
                else:
                    logging.info("img_id=[%s] load html successfully."%self.img_id)
                    is_opened = True
                    time.sleep(self.wait_second)
                    break
            if not is_opened:
                return -1
            html = response.read()
            self.save_html(html, cache_file)
        self.soup = BeautifulSoup(html, 'html.parser')
        self.request_num += 1
        return 0

    def save_html(self, html, cache_file):
        fhtml = open(cache_file, 'w')
        fhtml.write(html)
        fhtml.close()
        logging.info("[html]%s cached successfully."%cache_file)

    def get_datestamp(self):
        date_tag = self.soup.find("div", class_="data")
        if date_tag is None:
            return "NAN"
        else:
            return date_tag.string.strip()

    def get_title(self):
        title_tag = self.soup.find("div", class_="articleContent")
        if title_tag is None:
            return "NAN"
        else:
            return all_strip(title_tag.string)

    def get_zan_num(self):
        zan_num_tag = self.soup.find("span", id="zan-num")
        if zan_num_tag is None:
            return "-1"
        else:
            return zan_num_tag.string.strip()

    def get_cai_num(self):
        cai_num_tag = self.soup.find("span", id="cai-num")
        if cai_num_tag is None:
            return "-1"
        else:
            return cai_num_tag.string.strip()

    def get_view_num(self):
        article_data_tag = self.soup.find("div", class_="articleData")
        if article_data_tag is None:
            return "-1"
        for c in article_data_tag.contents:
            match = re.search(ur"(\d+)人浏览", c.string, flags=re.U)
            if match is not None:
                return match.group(1)
        return "-1"

    def get_hotness(self):
        article_data_tag = self.soup.find("div", class_="articleData")
        if article_data_tag is None:
            return "-1"
        for c in article_data_tag.contents:
            match = re.search(ur"(\d+\.?\d*)热度", c.string, flags=re.U)
            if match is not None:
                return match.group(1)
        return "-1"

    #def get_tag_list(self):
    #    pass

    def get_comment_list(self):
        comments_tag = self.soup.find("div", id="comment_content_all")
        if comments_tag is None:
            return []
        comment_list = []
        for comment in comments_tag.find_all("div", class_="comment"):
            try:
                nickname = all_strip(comment.find("a").string)
                content = all_strip(comment.find("div", class_="content").contents[-1])
                comment_list.append((nickname, content))
            except:
                continue
        return comment_list

    def save_image(self):
        img_url = self.get_img_url()
        if img_url == "":
            return 
        cached_img = self.img_path + "%s.jpg"%self.img_id
        if os.path.exists(cached_img):
            self.cached_img_num += 1
            logging.info("[img]%s has been cached."%cached_img)
        else:
            try:
                urllib.urlretrieve(img_url, cached_img)
            except Exception, e:
                logging.error("[img]%s cache failed."%img_url)
                logging.error(str(e))
            else:
                self.cached_img_num += 1
                logging.info("[img]%s cached successfully."%img_url)

    def load_prev_page(self):
        prev_page_href = self.get_prev_page_href()
        if prev_page_href == u"javascript:void(0);":
            loading.info("All Images has been cached!")
            self.is_loaded = False
            return 
        
        _idx = prev_page_href.find("&id=")
        if _idx == -1:
            self.img_id = str(int(self.img_id) + 1)
        else:
            self.img_id = prev_page_href[_idx+4:]
        if -1 == self.load_html():
            logging.fatal("loading html for img_id=[%s] failed!"%self.img_id)
            self.is_loaded = False
        else:
            self.is_loaded = True
        
    def get_img_url(self):
        img_tag = self.soup.find("img", id="imgSybj")
        if img_tag is None:
            return ""
        else:
            return img_tag.get("src", "").strip()
    
    def get_prev_page_href(self):
        prev_page_tag = self.soup.find("a", id="prepage")
        if prev_page_tag is None:
            return ""
        else:
            return prev_page_tag.get("href", "").strip()



if __name__ == "__main__":
    init_img_id = "313"
    max_request_num = 200000
    web_parser = WebParser(init_img_id, wait_second=1, max_retry_time=10)
    
    img_attr_csv_file = open("./data/img_attr.csv", 'aw')
    img_comments_file = open("./data/img_comments.csv", 'aw')

    while web_parser.is_loaded and web_parser.request_num <= max_request_num:
        img_id = web_parser.img_id
        try:
            datestamp = web_parser.get_datestamp()
            title = web_parser.get_title()
            zan_num = web_parser.get_zan_num()
            cai_num = web_parser.get_cai_num()
            view_num = web_parser.get_view_num()
            hotness = web_parser.get_hotness()
            #tags = web_parser.get_tag_list()
            
            img_attrs = "\t".join([img_id, zan_num, cai_num, view_num, hotness, datestamp, title]).encode("gbk")
            img_attr_csv_file.write(img_attrs+"\n")
        except Exception, e:
            logging.warning("img_id=[%s] attributes save failed."%img_id)
            logging.error(str(e))
        else:
            logging.info("img_id=[%s] attributes save successfully."%img_id)

        try:
            comment_list = web_parser.get_comment_list()
            for nickname, comment in comment_list:
                img_comments_file.write("\t".join([img_id, nickname, comment]).encode("gbk")+"\n")
        except Exception, e:
            logging.warning("img_id=[%s] comments save failed."%img_id)
            logging.error(str(e))
        else:
            logging.info("img_id=[%s] comments save successfully."%img_id)

        web_parser.save_image()

        web_parser.load_prev_page()

    logging.info("Done! Totally %d images has been cached successfully."%web_parser.cached_img_num)
    
    img_attr_csv_file.close()
    img_comments_file.close()
