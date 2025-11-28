The Rock Paper Scissors Game Created Using Hand Gesture Recognition (New Media Pipe and OpenCV) is a project that provides live video of the Rock Paper Scissors game.

This is accomplished through hand gesture recognition, using Media Pipe and OpenCV and running on Flask. The webcam captures the image of the player’s hands, recognizes the hands, and by measuring the location of the fingers, translates the hand gestures into Rock, Paper or Scissors in real-time.

The game has two ways of playing:

1. A way to play against another player using hand gestures (two-hand mode).

2. AI-generated opponent against whom the user competes (single-hand mode), which includes an automatic countdown timer before detecting the user’s moves.

Using the hands' landmark data from Media Pipe, the application determines whether each finger is extended or folded based on the finger's position in relation to the hand's orientation (right or left). Using logical reasoning, the user’s gestures are accurately detected.

A feature of the game allows for the countdown timer for the AI-generated opponent to be displayed without interfering with the video feed to the browser interface.

* Features

1)Live hand detection with MediaPipe

2)Accurate Rock–Paper–Scissors gesture classification

3)Two-player mode and AI opponent mode

4)Web-based live video feed (Flask streaming)

5)Non-blocking countdown timer

-----------------------------------------------------------

* Tech Stack

1)Python

2)OpenCV

3)MediaPipe

4)Flask

5)FPS display for performance monitoring

6)Works with standard webcam input
