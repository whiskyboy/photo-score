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
import requests
from multiprocessing import Process, Semaphore, Lock, Queue, Pool

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')

def loginfo(msg):
    logging.info(msg)
    
def logerror(msg):
    logging.error(msg)
    
def logwarning(msg):
    logging.warning(msg)

def all_strip(s):
    return "".join(s.split())

class WebParser(object):
    def __init__(self, wait_second=1, max_retry_time=10):
        self.html_cache_path = "./data/cache/"
        self.img_path = "./data/img/"
        self.img_id = -1
        self.soup = None
        self.wait_second = wait_second
        self.max_retry_time = max_retry_time
    
    def build_request_url(self):
        return "http://www.sybj.com/may.php?c=w&a=organizationCommunity&t=1&hid=1126&id=%s"%self.img_id
    
    def build_request_headers(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30"
        request_headers = {'User-Agent': user_agent}
        return request_headers

    def load_html(self, imgid):
        self.img_id = str(imgid)
        cache_file = self.html_cache_path + "%s.html"%self.img_id
        if os.path.exists(cache_file):
            loginfo("[imgid=%s]html has been cached."%self.img_id)
            html = open(cache_file, 'r').read()
        else:
            url = self.build_request_url()
            headers = self.build_request_headers()
            is_opened = False
            for _ in range(self.max_retry_time):
                try:
                    html = requests.get(url=url, headers=headers).content
                except Exception, e:
                    time.sleep(self.wait_second * 10)
                else:
                    loginfo("[imgid=%s]download html successfully."%self.img_id)
                    is_opened = True
                    break
            if not is_opened:
                logwarning("[imgid=%s]download html failed."%self.img_id)
                return -1
            self.save_html(html, cache_file)
        self.soup = BeautifulSoup(html, 'html.parser')
        return 0

    def save_html(self, html, cache_file):
        fhtml = open(cache_file, 'w')
        fhtml.write(html)
        fhtml.close()

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
        cached_img = self.img_path + "%s.jpg"%self.img_id
        if os.path.exists(cached_img):
            loginfo("[imgid=%s]image has been cached."%self.img_id)
        else:
            img_url = self.get_img_url()
            if img_url == "":
                logwarning("[imgid=%s]image does not exist."%self.img_id)
                return 
            try:
                urllib.urlretrieve(img_url, cached_img)
            except Exception, e:
                logwarning("[imgid=%s]image caches failed."%self.img_id)
            else:
                loginfo("[imgid=%s]image caches successfully."%self.img_id)

    def get_img_url(self):
        img_tag = self.soup.find("img", id="imgSybj")
        if img_tag is None:
            return ""
        else:
            return img_tag.get("src", "").strip()

if __name__ == "__main__":
    img_attr_csv_file = open("./data/img_attr.csv", 'a+')
    img_comments_file = open("./data/img_comments.csv", 'a+')
    
    # lock
    img_attr_lock = Lock()
    img_comment_lock = Lock()
    
    def func(imgid):
        web_parser = WebParser(wait_second=1, max_retry_time=10)
        if web_parser.load_html(imgid) != 0:
            return
        
        datestamp = web_parser.get_datestamp()
        title = web_parser.get_title()
        zan_num = web_parser.get_zan_num()
        cai_num = web_parser.get_cai_num()
        view_num = web_parser.get_view_num()
        hotness = web_parser.get_hotness()
            
        img_attrs = "\t".join([str(imgid), zan_num, cai_num, view_num, hotness, datestamp, title]).encode("gbk")
        try:
            img_attr_lock.acquire()
            img_attr_csv_file.write(img_attrs+"\n")
            img_attr_csv_file.flush()
        except Exception, e:
            logwarning("[imgid=%d]image attributes save failed."%imgid)
        finally:
            img_attr_lock.release()

        comment_list = web_parser.get_comment_list()
        try:
            img_comment_lock.acquire()
            for nickname, comment in comment_list:
                img_comments_file.write("\t".join([str(imgid), nickname, comment]).encode("gbk")+"\n")
            img_comments_file.flush()
        except Exception, e:
            logwarning("[imgid=%d]image comments save failed."%imgid)
        finally:
            img_comment_lock.release()

        web_parser.save_image()
    
    start_imgid = 170000
    end_imgid = 170100
    imgid_list = [imgid for imgid in range(start_imgid, end_imgid)]
    
    p = Pool()
    p.map_async(func, imgid_list)
    p.close()
    p.join()
    
    img_attr_csv_file.flush()
    img_comments_file.flush()
    img_attr_csv_file.close()
    img_comments_file.close()
    
    loginfo("Finished.")
    
