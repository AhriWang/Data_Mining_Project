import os
import gzip
import requests
import pandas as pd
from tqdm import tqdm
from lxml import etree

# ====== 配置参数 ======
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
SAVE_DIR = "pubmed_data"
NUM_FILES = 3  # 控制下载几个文件，建议先 3 个试试
OUTPUT_CSV = "pubmed_titles_abstracts.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 获取 .xml.gz 文件列表 ======
def get_file_list():
    index_url = BASE_URL
    response = requests.get(index_url)
    file_list = []
    for line in response.text.splitlines():
        if '.xml.gz' in line:
            start = line.find('href="') + len('href="')
            end = line.find('.xml.gz') + len('.xml.gz')
            filename = line[start:end]
            if filename.endswith('.xml.gz'):
                file_list.append(filename)
    return file_list

# ====== 下载 .gz 文件 ======
def download_file(filename):
    url = BASE_URL + filename
    local_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} ...")
        with requests.get(url, stream=True) as r:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_path

# ====== 使用 lxml 提取 Title 和 Abstract ======
def extract_articles_from_gz_lxml(path):
    articles = []
    with gzip.open(path, 'rb') as f:
        tree = etree.parse(f)
        root = tree.getroot()
        for article in root.xpath('.//PubmedArticle'):
            try:
                title_nodes = article.xpath('.//ArticleTitle')
                abstract_nodes = article.xpath('.//Abstract/AbstractText')

                title = title_nodes[0].text if title_nodes else None
                abstract = ' '.join([
                    node.text.strip() for node in abstract_nodes if node.text
                ]) if abstract_nodes else None

                if title and abstract:
                    articles.append({
                        'title': title.strip(),
                        'abstract': abstract.strip()
                    })
            except Exception as e:
                continue
    return articles

# ====== 主函数 ======
def main():
    all_articles = []
    file_list = get_file_list()

    for filename in tqdm(file_list[:NUM_FILES], desc="Processing files"):
        print(f"\n>>> Processing: {filename}")
        gz_path = download_file(filename)
        articles = extract_articles_from_gz_lxml(gz_path)
        print(f"Extracted {len(articles)} articles.")
        all_articles.extend(articles)

    df = pd.DataFrame(all_articles)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 完成！共提取 {len(df)} 条记录，已保存至：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
