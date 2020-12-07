from multiprocessing import Pool,Process
# from inference_0 import main_0
from inference_1 import main_1


if __name__ == '__main__':
    p0 = Process(target=main_0)
    p1 = Process(target=main_1)
    p0.start()
    p1.start()
    

