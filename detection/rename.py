import os

def rename_images(folder_path):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    for i, filename in enumerate(image_files, start=1):
        _, ext = os.path.splitext(filename)
        new_filename = f"img{i}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

folder_path = "dataset"
rename_images(folder_path)
