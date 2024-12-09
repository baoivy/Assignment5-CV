from panaroma import image_stitch
import imutils
import cv2

filename = ['image/image1.jpg', 'image/image2.jpg', 'image/image3.jpg']

no_of_images = len(filename)    
images = []
for i in range(len(filename)):
    images.append(cv2.imread(filename[i]))

# We need to modify the images width and height to keep our aspect ratio same across images
for i in range(len(filename)):
    images[i] = imutils.resize(images[i], width=400, height=400)

if no_of_images == 2:
    result = image_stitch([images[0], images[1]])
else:
    result = image_stitch([images[no_of_images - 2], images[no_of_images - 1]])
    for i in range(no_of_images - 2):
        result = image_stitch([images[no_of_images - i - 3], result])

cv2.imshow("Panorama", result)
cv2.imwrite("output/panorama_image.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()