import cv2
from src.preprocessing.pipeline import preprocess_image

image_path = "data/raw/prescriptions/images/1.jpg"

processed = preprocess_image(image_path)

cv2.imshow("Processed Image", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()

