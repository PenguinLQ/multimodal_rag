import os
from multimodal_rag.common.config import get_settings

settings = get_settings()

raw_dir = settings.data_dir + "/raw"
filename_list = [raw_dir+"/"+f for f in os.listdir(raw_dir)]

for f in filename_list:
    print(f)
