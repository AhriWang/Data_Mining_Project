import os
import json
from bs4 import BeautifulSoup
import re


def extract_text_and_title_from_html(html_filepath):
    """
    从指定的 HTML 文件中提取标题和正文文本。
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '')

        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content')
        if not content_tag:
            content_tag = soup.find('article')
        if not content_tag:
            content_tag = soup.find('main')
        if not content_tag:
            content_tag = soup.find('body')

        if content_tag:
            text = content_tag.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n', text).strip()
            text = text.replace('阅读原文', '').strip()
            return title, text
        else:
            print(f"警告：在文件 {html_filepath} 中未找到明确的正文标签。")
            return title, None

    except FileNotFoundError:
        print(f"错误：文件 {html_filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"处理文件 {html_filepath} 时出错: {e}")
        return None, None


def split_text_by_sentence(text, chunk_size=512, chunk_overlap=50):
    """
    按句子边界切分文本为块，每块不超过 chunk_size，可重叠 chunk_overlap。
    """
    if not text:
        return []

    # 1. 用正则分句
    sentences = re.split(r'(?<=[。！？!?\.])\s*', text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            # 控制重叠（尾部保留）
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# --- 配置 ---
html_directory = './data/'
output_json_path = './data/processed_data.json'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- 主逻辑 ---
all_data_for_milvus = []
file_count = 0
chunk_count = 0

print(f"开始处理目录 '{html_directory}' 中的 HTML 文件...")

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
html_files = [f for f in os.listdir(html_directory) if f.endswith('.html')]
print(f"找到 {len(html_files)} 个 HTML 文件。")

for filename in html_files:
    filepath = os.path.join(html_directory, filename)
    print(f"  处理文件: {filename} ...")
    file_count += 1

    title, main_text = extract_text_and_title_from_html(filepath)

    if main_text:
        chunks = split_text_by_sentence(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"    提取到文本，分割成 {len(chunks)} 个块。")

        for i, chunk in enumerate(chunks):
            chunk_count += 1
            milvus_entry = {
                "id": f"{filename}_{i}",
                "title": title or filename,
                "abstract": chunk,
                "source_file": filename,
                "chunk_index": i
            }
            all_data_for_milvus.append(milvus_entry)
    else:
        print(f"    警告：未能从 {filename} 提取有效文本内容。")

print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个文本块。")

# --- 保存 JSON ---
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_milvus, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_json_path}")
except Exception as e:
    print(f"错误：无法写入 JSON 文件 {output_json_path}: {e}")
