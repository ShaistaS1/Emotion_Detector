import cv2
import torch
import os
import sys
from model import EmotionCNN, initialize_model
from torchvision import transforms

def find_model():
    """Search the entire project for the model file"""
    # Get the project root (2 levels up from src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Possible model locations
    search_paths = [
        os.path.join(project_root, 'data', 'models', 'best_model.pth'),
        os.path.join(project_root, 'models', 'best_model.pth'),
        os.path.join(project_root, 'best_model.pth')
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    # If not found, show help
    print("\nERROR: Model file not found in these locations:")
    for path in search_paths:
        print(f"- {path}")
    print("\nSOLUTIONS:")
    print("1. First train your model: python src/train.py")
    print("2. Or manually place best_model.pth in data/models/")
    sys.exit(1)

class EmotionDetector:
    def __init__(self):
        model_path = find_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = initialize_model(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            tensor = self.transform(face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(tensor)
                emotion = self.emotions[torch.argmax(pred).item()]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame

def main():
    detector = EmotionDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not accessible")
        return
    
    print("Press Q to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame = detector.detect(frame)
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()