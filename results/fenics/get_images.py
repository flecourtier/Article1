import os
import zipfile

def get_images_and_archive(root_dir, archive_name):
    with zipfile.ZipFile(archive_name, 'w') as archive:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    file_path = os.path.join(foldername, filename)
                    archive_path = os.path.relpath(file_path, root_dir)
                    archive.write(file_path, archive_path)

if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))
    archive_name = 'images_archive.zip'
    get_images_and_archive(root_directory, archive_name)