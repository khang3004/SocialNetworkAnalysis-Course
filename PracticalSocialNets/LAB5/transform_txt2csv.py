import csv

# Đường dẫn tệp gốc và tệp đích
input_file = "./YouTube_Social_Network_Data.txt"  # Thay bằng tên tệp .txt của bạn
output_file = "./YouTube_Social_Network_Data.csv"   # Tên tệp .csv sau khi chuyển đổi

# Đọc tệp .txt và ghi ra tệp .csv
with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
    reader = txt_file.readlines()  # Đọc tất cả dòng từ tệp .txt
    writer = csv.writer(csv_file)
    
    # Ghi tiêu đề cho tệp .csv
    writer.writerow(["Source", "Target"])
    
    # Bỏ qua các dòng bắt đầu bằng "#" và xử lý các dòng còn lại
    for line in reader:
        if not line.startswith("#"):  # Bỏ qua các dòng bình luận
            parts = line.strip().split()  # Tách các cột bằng dấu cách
            if len(parts) == 2:  # Đảm bảo mỗi dòng có 2 phần
                writer.writerow(parts)  # Ghi dòng vào tệp .csv

print(f"File đã được chuyển đổi thành công sang: {output_file}")
