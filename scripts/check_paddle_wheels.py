import urllib.request
import re
import json

def check_paddle_support():
    print("=== Checking PaddlePaddle and PaddleOCR Support on Windows ===")
    
    # Check official PyPI index metadata for paddlepaddle
    urls = [
        "https://pypi.tuna.tsinghua.edu.cn/simple/paddlepaddle/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/paddleocr/"
    ]
    
    for url in urls:
        pkg_name = url.strip("/").split("/")[-1]
        print(f"\n--- Checking package: {pkg_name} ---")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pip/24.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8")
            
            # Find wheels
            wheels = re.findall(r'>([^<]+\.whl)<', html)
            print(f"Total wheels listed: {len(wheels)}")
            
            win_wheels = [w for w in wheels if "win_amd64" in w]
            print(f"Windows amd64 wheels: {len(win_wheels)}")
            
            cp_versions = set()
            for w in win_wheels:
                matches = re.findall(r'cp(3\d+)', w)
                for m in matches:
                    cp_versions.add(f"Python 3.{m[1:]}")
            
            print(f"Supported Python versions on Windows: {sorted(list(cp_versions))}")
            
            # Check latest versions
            print("Sample latest Windows wheels:")
            for w in win_wheels[-5:]:
                print(f"  - {w}")
                
        except Exception as e:
            print(f"Error checking {url}: {e}")

if __name__ == '__main__':
    check_paddle_support()
