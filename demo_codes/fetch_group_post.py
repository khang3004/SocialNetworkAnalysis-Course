from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Cài đặt trình duyệt (Chrome)
driver = webdriver.Chrome()

# Đăng nhập LinkedIn
driver.get("https://www.linkedin.com/login")
time.sleep(2)

# Nhập thông tin đăng nhập
username = driver.find_element(By.ID, "username")
password = driver.find_element(By.ID, "password")

username.send_keys("your_email")
password.send_keys("your_password")
password.send_keys(Keys.RETURN)
time.sleep(5)

# Truy cập trang tìm kiếm công khai
search_url = "https://www.linkedin.com/search/results/people/?keywords=data+scientist"
driver.get(search_url)
time.sleep(5)

# Thu thập tên người dùng
profiles = driver.find_elements(By.CLASS_NAME, "entity-result__title-text")
for profile in profiles:
    print(profile.text)

# Đóng trình duyệt
driver.quit()
