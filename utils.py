import pytesseract
import cv2
import easyocr

# Set the Tesseract command path if needed
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path to your system

# Initialize EasyOCR Reader for English
reader = easyocr.Reader(['en'], gpu=False)

def ocr_core(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use EasyOCR first for detecting text (especially for handwriting)
    results_easyocr = reader.readtext(image)
    
    # Draw the EasyOCR results on the image
    for (bbox, text, confidence) in results_easyocr:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw bounding box and the text
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Use Tesseract as a fallback for detecting text that EasyOCR might miss (especially for printed text)
    data_tesseract = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Iterate through each detected text and its bounding box
    for i in range(len(data_tesseract['text'])):
        if int(data_tesseract['conf'][i]) > 60:  # Only consider high-confidence text
            x, y, w, h = data_tesseract['left'][i], data_tesseract['top'][i], data_tesseract['width'][i], data_tesseract['height'][i]
            text = data_tesseract['text'][i]
            
            # Draw bounding box and text for Tesseract results
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    return image
