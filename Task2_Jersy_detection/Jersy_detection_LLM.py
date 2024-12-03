import cv2
import easyocr
import openai
from ultralytics import YOLO  # Import YOLO from the Ultralytics library

# Set your OpenAI API key
openai.api_key = "sk-proj-YYWEi9jOvto3pAHuQYn7T3BlbkFJniOvU7p7wxHC10Liulu8"

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Load your YOLO model
model = YOLO("runs/detect/hockey_detector/weights/best.pt")  # Path to your YOLO model

# Function to refine OCR output using OpenAI GPT
def refine_ocr_output(ocr_text, context):
    try:
        prompt = (
            f"The following text is extracted from a jersey number detection system: '{ocr_text}'. "
            f"Context: {context}. Clean the text and return only the correct jersey number. "
            f"If the number is not visible or unclear, respond with 'Unknown'."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an OCR output cleaner for jersey number detection."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return "Unknown"

# Function to extract jersey numbers using EasyOCR
def extract_jersey_number(image, context):
    try:
        # Run OCR on the cropped image
        results = reader.readtext(image, detail=0)  # Extract text without bounding box details
        ocr_text = " ".join(results).strip() if results else "Unknown"
        
        # Refine OCR output using OpenAI GPT
        jersey_number = refine_ocr_output(ocr_text, context)
        return jersey_number
    except Exception as e:
        print(f"Error with EasyOCR: {e}")
        return "Unknown"

# Function to process video and detect jersey numbers
def process_video(video_path, context, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames for a 5-second clip
    duration = 5  # seconds
    max_frames = min(total_frames, fps * duration)

    # Video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model to detect players
        results = model.predict(frame, conf=0.5)  # Adjust confidence threshold if needed
        detections = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])  # Bounding box coordinates and class
            
            # Crop the player region for jersey number detection
            player_roi = frame[y1:y2, x1:x2]

            # Extract jersey number using EasyOCR and refine with GPT
            jersey_number = extract_jersey_number(player_roi, context)

            # Draw bounding box and display the detected number above the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, jersey_number, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

# Run the program
if __name__ == "__main__":
    video_path = r"C:\Users\KIIT\Desktop\MEDI project\Task2\Testing_vid.mp4"  # Path to your input video
    output_path = r"C:\Users\KIIT\Desktop\MEDI project\Task2\output_detected_players.avi"  # Path to save output video
    context = "This is a hockey match. Detect jersey numbers from the back of the players."

    process_video(video_path, context, output_path)
