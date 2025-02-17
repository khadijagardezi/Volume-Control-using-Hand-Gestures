# Volume Control using Hand Gestures
This project introduces a system for controlling volume of the computer
system using hand gestures as input. The system utilizes the OpenCV
module for gesture detection and control. It captures images and videos
through a webcam and adjusts the system’s volume based on user
gestures.

- The system detects hand gestures in real-time, translating them directly into volume up/down commands.
- It uses a webcam / sensor for raw data capture, followed by signal processing and segmentation for accurate gesture classification.
- Eliminates the need for physical buttons or knobs, offering a touch-free experience
- The underlying gesture recognition framework can be extended to control additional features, such as track selection or playback.

## Steps

- Use camera or sensor to capture hand gestures
- Identify the input by user
- MediaPipe determines appropriate volume level based on the user's gesture
- Volume information sent to the device
- Device adjusts the volume level accordingly


## Applications of Gesture Recognition System

- Talking to the Computer
- Medical Operations
- Gesture-based Gaming Control
- Hand Gesture to control the home appliances like MP3 player, TV etc.
- Gesture Control Car Driving
- Communication


## Screenshots

![App Screenshot](https://github.com/khadijagardezi/Volume-Control-using-Hand-Gestures/blob/main/gesture.png)
![App Screenshot](https://github.com/khadijagardezi/Volume-Control-using-Hand-Gestures/blob/main/image-1.png)
![App Screenshot](https://github.com/khadijagardezi/Volume-Control-using-Hand-Gestures/blob/main/image-2.png)

## Future Work

- Mobile implementations TensorFlow Lite or MediaPipe's mobile SDK for real-time processing. 
- Add multiple gesture detection for play/pause, next/previous track.
- Use AI models to train custom gestures.
- Implement voice feedback for accessibility.


## Documentation

[Documentation](https://drive.google.com/file/d/1tbSpUKQPXjCojrHkD3vg9Yz412JQ6ppn/view?usp=sharing)




