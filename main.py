import face_recognition
import cv2
import os

# Set the name of the test image you want to recognize faces in
TEST_IMAGE_NAME = "/test.jpg"

# Lists to store known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Directory containing the known face images
known_faces_dir = "./known_faces"

# Loop through each file in the known faces directory
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        # Check if at least one face encoding was found in the image
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])  # Store the first face encoding
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as the name

# Load the image where we want to recognize faces
test_image = face_recognition.load_image_file("./test_image" + TEST_IMAGE_NAME)

# Detect all face locations in the test image
face_locations = face_recognition.face_locations(test_image)

# Get encodings for each detected face
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert the image from RGB to BGR (OpenCV uses BGR format)
image_cv = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

# Loop through each detected face and its encoding
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compare this face to all known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"  # Default label

    # Compute the distance (similarity) between this face and known faces
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    # Find the best match (smallest distance)
    best_match_index = face_distances.argmin() if face_distances.size > 0 else -1

    # If the best match is a confirmed match, use the known name
    if best_match_index >= 0 and matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw the name label above the face
    cv2.putText(image_cv, name, (left + 6, top - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    
    # Draw a green rectangle around the detected face
    cv2.rectangle(image_cv, (left - 15, top - 15), (right + 15, bottom + 15), (0, 255, 0), 2)

# Show the final image with labeled faces
cv2.imshow("Face Recognition", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
