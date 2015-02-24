__author__ = 'kiruba'
from mechanize import Browser
from BeautifulSoup import BeautifulSoup
import urlparse
import re
import urllib2
import socket
import os
import errno
import time

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

mech = Browser()
url = "http://octamil.com/Tamil%20Download/21%20-%205.1%20Surround%20Albums/All%20Movies%20Flac%20Collection/"
page = mech.open(url)
html = page.read()
soup = BeautifulSoup(html)
list_url = []
for tag in soup.findAll('a', href=True):
    tag['href'] = urlparse.urljoin(url, tag['href'])
    list_url.append(tag['href'])

mystring = '/21 - 5.1 Surround Albums/All Movies Flac Collection/'
indices  = [i for i, x in enumerate(list_url) if re.search(mystring, x)]
album_url = [list_url[i] for i in indices]
del album_url[-1]
# print album_url
song_url = []
songs_folder = '/media/kiruba/B21E7CD21E7C9159/FLAC/'
class MyException(Exception):
    pass

for album in album_url[8:9 ]:
    album = album.replace(u'%20', u' ')
    album_regex = album + '/'
    album_name = album.split('/')[-1]
    print album_name
    # raise SystemExit(0)
    album_page = mech.open(album)
    album_html = album_page.read()
    album_soup = BeautifulSoup(album_html)
    for tag in album_soup.findAll('a', href=True):
        link = urlparse.urljoin(url, tag['href'])
        if re.match(album_regex, link):
            file_name = link.split('/')[-1]
            file_name = file_name.replace(u'%20', u' ')
            # raise SystemExit(0)
            try:
                u = urllib2.urlopen(link, timeout=1)
            except urllib2.URLError as e:
                print type(e)
            except socket.timeout as e:
                print type(e)
                raise MyException("There was an error: %r" % e)
            folder_name = songs_folder + album_name
            make_sure_path_exists(folder_name)
            with open(songs_folder+album_name+ '/' + file_name, 'wb') as f:
                meta = u.info()
                file_size = int(meta.getheaders("Content-Length")[0])
                print "Downloading: %s Bytes: %s" % (file_name, file_size)
                file_size_dl = 0
                block_sz = 60*1024
                while True:
                    buffer = u.read(block_sz)
                    if not buffer:
                        break

                    file_size_dl += len(buffer)
                    f.write(buffer)
                    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                    status = status + chr(8)*(len(status)+1)
                    print status,
            time.sleep(60)

# print song_url
# class MyException(Exception):
#     pass
# songs_folder = '/media/kiruba/B21E7CD21E7C9159/FLAC/'
# for song in song_url:
#     file_name = song.split('/')[-1]
#     try:
#         u = urllib2.urlopen(song, timeout=1)
#     except urllib2.URLError as e:
#         print type(e)
#     except socket.timeout as e:
#         print type(e)
#         raise MyException("There was an error: %r" % e)
#     f = open(songs_folder+file_name, 'wb')
#     meta = u.info()
#     file_size = int(meta.getheaders("Content-Length")[0])
#     print "Downloading: %s Bytes: %s" % (file_name, file_size)
#     file_size_dl = 0
#     block_sz = 8192
#     while True:
#         buffer = u.read(block_sz)
#         if not buffer:
#             break
#
#         file_size_dl += len(buffer)
#         f.write(buffer)
#         status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
#         status = status + chr(8)*(len(status)+1)
#         print status,
#
# f.close()
