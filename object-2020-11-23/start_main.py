import time
from multiprocessing import Pool,Process
from inference_0 import main_0
from inference_1 import main_1
from inference_trip import get_trip_class  


if __name__ == '__main__':
    time.sleep(3)
    trip_class = get_trip_class()
    if trip_class == 0:
        p0 = Process(target=main_0)
        p1 = Process(target=main_1)
        p0.start()
        p1.start()
    else:
        print("弹条的识别结果是三型！")

    
