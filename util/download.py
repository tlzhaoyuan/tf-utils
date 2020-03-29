from urllib.request import urlopen


def download(url, dst):
    # Upload data from GitHub to notebook's local drive
    response = urlopen(url)
    html = response.read()
    with open(dst, 'wb') as fp:
        fp.write(html)