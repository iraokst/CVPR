import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Can't open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("Can't receive frame")
    exit()

cv2.imwrite('image.jpg', frame)
img = cv2.imread('image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

start_point_rectangle = (100, 30)
end_point_rectangle = (300, 250)
color_rectangle_bgr = (41, 241, 248)

start_point_line = (15, 20)
end_point_line = (450, 300)
color_line_bgr = (97, 41, 248)

gray_rectangle = cv2.rectangle(gray, start_point_rectangle, end_point_rectangle, color_rectangle_bgr, -1)
gray_rectangle_line = cv2.line(gray_rectangle, start_point_line, end_point_line, color_line_bgr, 3)

cv2.imshow('frame', frame)
cv2.imshow('gray_frame', gray_rectangle_line)

cv2.imwrite('gray_image.jpg', gray_rectangle_line)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()