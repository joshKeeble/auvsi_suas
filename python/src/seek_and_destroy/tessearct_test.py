from PIL import Image
import pytesseract
import cv2

frame = cv2.imread("/home/hal/Desktop/Programming/generated_data/letters/A/circle_A_2.jpg")

text = pytesseract.image_to_string(frame,lang = 'eng')

print('Text:{}'.format(text))

cv2.imshow("frame",frame)
cv2.waitKey(0)