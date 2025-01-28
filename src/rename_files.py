# import os

# for root, dirs, files in os.walk("."):
#     for name in files:
#         print(name)
#         if "v7" in name:
#             old_path = os.path.join(root, name)
#             new_path = os.path.join(root, name.replace("v7", "v2"))
#             try:
#                 os.rename(old_path, new_path)
#                 print(f"File renamed: {old_path} -> {new_path}")
#             except Exception as e:
#                 print(f"Error renaming file {old_path}: {e}")
#     for name in dirs:
#         if "v7" in name:
#             old_path = os.path.join(root, name)
#             new_path = os.path.join(root, name.replace("v7", "v2"))
#             try:
#                 os.rename(old_path, new_path)
#                 print(f"Directory renamed: {old_path} -> {new_path}")
#             except Exception as e:
#                 print(f"Error renaming directory {old_path}: {e}")
