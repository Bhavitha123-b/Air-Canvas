# Air Canvas – Gesture-Based Drawing System

An interactive **gesture-controlled drawing application** built using **Python, OpenCV, and MediaPipe**.
The system allows users to draw in the air using hand gestures detected through a webcam. It supports dynamic brush control, automatic shape detection, object manipulation, and a floating UI palette.

This project demonstrates **computer vision, real-time gesture recognition, and interactive UI design using OpenCV**.

---

# Features

* ✋ **Hand Gesture Drawing** – Draw on the screen using your index finger.
* 🎨 **Dynamic Brush Thickness** – Brush size adjusts based on finger distance.
* 🧽 **Palm Eraser Mode** – Open palm gesture activates erase mode.
* 🔷 **Automatic Shape Detection**

  * Line
  * Circle
  * Triangle
  * Rectangle / Square
  * Ellipse
  * Pentagon / Hexagon / Polygon
* 🧩 **Object Selection and Dragging** – Move detected shapes around the canvas.
* ↩ **Undo / Redo System** – Revert or restore drawing actions.
* 💾 **Canvas Saving** – Save the final drawing as an image.
* 🖌 **Floating UI Palette** – Minimal UI for color selection and tool controls.
* ✨ **Smooth Drawing** – Moving average filtering reduces jitter while drawing.

---

# Demo

The application allows users to interact with a digital canvas using only hand gestures.

Example interactions:

* Draw freely in the air
* Select drawing colors using gestures
* Detect and convert strokes into geometric shapes
* Select shapes and move them on the canvas
* Erase parts of the drawing using palm gesture
* Save the canvas as an image

*(You can add screenshots or a demo video here)*

---

# Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe**
* **NumPy**
* **Deque (for smoothing filters)**

---

# Project Structure

Currently the project is implemented in a single file:

```
air.py
```

Future improvements may split the project into modules like:

```
air_canvas/
│
├── main.py
├── gestures.py
├── shape_detection.py
├── ui.py
├── config.py
└── utils.py
```

---

# Installation

## 1. Clone the repository

```
git clone https://github.com/Bhavitha123-b/air-Canvas.git
cd air-Canvas
```

## 2. Install dependencies

```
pip install opencv-python mediapipe numpy
```

## 3. Run the application

```
python air.py
```

---

# Controls & Gestures

| Gesture                               | Action                               |
| ------------------------------------- | ------------------------------------ |
| **Index Finger Up**                   | Draw on the canvas                   |
| **Index + Middle Finger Up**          | Select color from palette            |
| **Index + Middle + Ring Finger Up**   | Select an object/shape on the canvas |
| **Move Index Finger After Selection** | Drag / move the selected object      |
| **Open Palm**                         | Activate eraser mode                 |

---

# Shape Detection Method

Shapes are detected using **OpenCV contour analysis**:

* `cv2.findContours()` extracts stroke contours.
* `cv2.approxPolyDP()` approximates the contour to detect vertices.
* Geometric heuristics classify shapes:

  * **Aspect Ratio** → rectangle vs square
  * **Circularity** → circle detection
  * **Compactness** → ellipse classification
  * **Vertex Count** → triangle, pentagon, hexagon, polygon

---

# Applications

* Gesture-based UI systems
* Touchless interfaces
* Smart whiteboards
* Educational tools
* Computer vision learning projects

---

# Future Improvements

* Modular project structure
* AI-based shape recognition
* Multi-user gesture interaction
* Gesture-based UI menus
* Export drawings as vector graphics

---

# Author

Developed as a **Computer Vision project using OpenCV and MediaPipe** to explore real-time gesture interaction systems.

---

# License

This project is open-source and available under the **MIT License**.
