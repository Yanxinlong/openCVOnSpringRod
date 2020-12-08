from inference_0 import ser_0
from inference_1 import ser_1

def test_com_open(ser_0,ser_1):
	if ser_0.isOpen():
		print("Open ser_0 success!")
	else:
		print("Open ser_0 failed!")

	if ser_1.isOpen():
		print("Open ser_1 success!")
	else:
		print("Open ser_1 failed!")

	return ser_0.isOpen() and ser_1.isOpen()


def test_com_close(ser_0,ser_1):
	ser_0.close()
	ser_1.close()

	if not ser_0.isOpen():
		print("Close ser_0 success!")
	else:
		print("Close ser_0 failed!")

	if not ser_1.isOpen():
		print("Close ser_1 success!")
	else:
		print("Close ser_1 failed!")


if __name__ == '__main__':
	state = test_com_open(ser_0,ser_1)
	if state:
		test_com_close(ser_0,ser_1)