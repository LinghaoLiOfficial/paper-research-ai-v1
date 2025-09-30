import json
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from tqdm import tqdm
import sys
import urllib.parse
from requests.exceptions import ConnectTimeout



class RequestSender:
    @classmethod
    def post(cls, url: str, payload: dict):
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            message = data['choices'][0]['message']['content']
            return message

        return None

    @classmethod
    def download_pdf_file(cls, pdf_url, file_path, retries=5):  # 增加重试次数

        """
            针对遇到的下载PDF时的404和403错误的解决方案：
            (1)动态设置Referer头：根据PDF链接的域名动态生成Referer，确保与目标网站匹配
            (2)添加更真实的浏览器请求头：模拟更真实的浏览器请求，减少被识别为爬虫的风险
            (3)处理可能得重定向和会话：确保正确处理重定向，并保持会话一致性
            (4)处理特定网站的访问权限：对于需要登录的网站（如IEEE Xplore），需提供认证信息：
                # 示例：添加cookies（需先获取有效cookies）
                    cookies = {'session_id': 'your_session_cookie'}
                    response = session.get(
                        pdf_url,
                        headers=headers,
                        cookies=cookies,
                        # 其他参数
                    )
        """

        message = ""

        # 在download_pdf_file方法中修改headers部分
        parsed_url = urllib.parse.urlparse(pdf_url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': referer,
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',
        }

        session = requests.Session()

        # 增强重试策略（增加对连接错误的处理）
        retry = Retry(
            total=retries,
            connect=0,  # 单独禁用连接错误重试
            backoff_factor=1,  # 增加退避因子
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST']),  # 明确允许的方法
            respect_retry_after_header=True  # 遵守服务器的重试要求
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=50,
            pool_maxsize=50,
        )
        session.mount('https://', adapter)

        try:
            # 增加超时时间并禁用SSL验证
            with session.get(pdf_url,
                             headers=headers,
                             stream=True,
                             verify=False,  # 关闭SSL验证（注意安全风险）
                             timeout=(20, 60),
                             allow_redirects=True) as response:  # 延长超时时间

                response.raise_for_status()

                # 获取文件大小用于进度显示
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size,
                                    desc=os.path.basename(file_path),
                                    unit='B',
                                    unit_scale=True)

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=16384):  # 增大块大小
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()

                message = f"PDF 下载成功: {pdf_url}"

                return True, message

        except Exception as e:
            print(f"Download failed: {str(e)}")
            message = f"PDF {pdf_url} Download failed: {str(e)}"
            if os.path.exists(file_path):
                os.remove(file_path)
            return False, message

    # 定义一个函数来发送请求并处理重试逻辑
    @classmethod
    def send_request_with_retry(cls, url, params, max_retries=6):
        retries = 0  # 初始化重试计数器
        while retries < max_retries:
            try:
                response = requests.get(url, params=params)

                if response.status_code == 200:
                    return {
                        "check": True,
                        "message": "成功",
                        "data": response.json()
                    }
                else:
                    retries += 1  # 增加重试次数
                    time.sleep(10)  # 等待重试
            except requests.exceptions.RequestException as e:
                retries += 1  # 增加重试次数
                time.sleep(10)  # 等待重试
        # 如果达到最大重试次数仍未成功
        raise Exception(f"请求失败，已达到最大重试次数 {max_retries} 次。")


