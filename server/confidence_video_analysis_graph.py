import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Load the trained model
model = tf.keras.models.load_model('model/confidence_model_Adam.h5')  # Change model name if needed

# Define image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64

def predict_confidence(frame, model):
    frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    frame = np.expand_dims(frame, axis=0) / 255.0
    prediction = model.predict(frame)
    return prediction[0][0] * 100  # Convert to percentage

# Capture video from file or webcam
cap = cv2.VideoCapture('D:/projects/semester projects/phase 2 project sem 8/final-react-projecgt/final-react-projecgt/server/uploads/video.mp4')  # Change path for a different video file

confidence_levels = []
timestamps = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    confidence = predict_confidence(frame, model)
    confidence_levels.append(confidence)
    timestamps.append(time.time() - start_time)
    
    cv2.putText(frame, f'Confidence: {confidence:.2f}%', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Confidence Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Compute average confidence
average_confidence = np.mean(confidence_levels)

# Plot confidence graph
plt.figure(figsize=(10, 5))
plt.plot(timestamps, confidence_levels, label='Confidence')
plt.xlabel('Time (s)')
plt.ylabel('Confidence Level (%)')
plt.title('Results of emotions during interview')
plt.legend()
plt.show()

# Display final average confidence
print(f'Average Confidence during the interview was {average_confidence:.2f}%')
