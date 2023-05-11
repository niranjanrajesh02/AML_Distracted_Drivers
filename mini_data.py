import pandas as pd
import numpy as np
import os

DATA_PATH = "C:\Niranjan\Ashoka\Semester 6\AML\FinalProject\Data"

data_df = pd.read_csv(os.path.join(DATA_PATH, 'driver_imgs_list.csv'))