import cv2

# Load the cascade classifier
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Load image from local device
img = cv2.imread("license1.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect license plates
plates = plate_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(150,40)
)

# Draw rectangles around detected plates
for i,(x,y,w,h) in enumerate(plates, start=1):

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(img,
                f"Plate {i}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    # Crop detected plate
    plate_roi = img[y:y+h, x:x+w]
    cv2.imshow(f"Plate {i}", plate_roi)

# Show result
cv2.imshow("Detected License Plate", img)

cv2.waitKey(0)
cv2.destroyAllWindows()