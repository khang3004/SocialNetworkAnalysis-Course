from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import getpass
from datetime import datetime
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class FacebookDataScrapper:
    def __init__(self):
        self.driver = self.setup_driver()
        
    def setup_driver(self):
        """Thiết lập trình điều khiển Chrome"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        except Exception as e:
            print(f"Lỗi khi khởi tạo driver: {e}")
            return None
            
    def login(self, email, password):
        """Đăng nhập vào Facebook"""
        try:
            self.driver.get("https://www.facebook.com")
            
            wait = WebDriverWait(self.driver, 20)
            
            email_field = wait.until(
                EC.presence_of_element_located((By.ID, "email"))
            )
            password_field = wait.until(
                EC.presence_of_element_located((By.ID, "pass"))
            )
            
            email_field.clear()
            password_field.clear()
            
            email_field.send_keys(email)
            password_field.send_keys(password)
            
            login_button = wait.until(
                EC.element_to_be_clickable((By.NAME, "login"))
            )
            login_button.click()
            
            time.sleep(5)
            return True
        except Exception as e:
            print(f"Lỗi khi đăng nhập: {e}")
            return False
    def get_group_members(self):
        try:
            self.driver.get(f'https://www.facebook.com/groups/{self.group_id}/members/')
            time.sleep(6)
            members = set() 
            
            for i in range(self.scroll_count):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                print(f'Scroll lần {i+1}/{self.scroll_count}')

                user_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/user/']")
                print(len(user_elements))

                for user in user_elements:
                    try: 
                        href = user.get_attribute('href')
                        if '/user/' in href:
                            user_id = href.split('/')[1].strip('/')
                            name = user.text
                            members.add((user_id, name))
                            print(user_id, "-", name)
                    except Exception as e:
                        continue
            return list(members)

        except Exception as e:
            print(f'{e}')
def main():
    print("\n=====FACEBOOK GROUP MEMBER SCRAPPER ====")
    
    scraper = FacebookDataScrapper()
    if not scraper.driver:
        print("Không thể khởi tạo trình duyệt. Vui lòng kiểm tra lại ChromeDriver.")
        return
    
    try:
        print("Nhap thong tin dang nhap:")
        email = input("Email/Username: ")
        password = getpass.getpass("Password: ")
        
        if scraper.login(email, password):
            print("Đăng nhập thành công!")
            
            print("\nNhap ID group facebook:")
            group_id = input("Group ID: ")
            
            print("\nSo lan scroll de load members")
            scroll_count = int(input("Scroll count: "))
            
        else:
            print("Đăng nhập thất bại!")
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
    
    finally:
        if scraper.driver:
            scraper.driver.quit()

if __name__ == "__main__":
    main()