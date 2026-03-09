# Air Canvas: A Step-by-Step Project Explanation

Welcome to the **Air Canvas** project! This guide will explain everything about this project from the ground up. Whether you are a beginner looking to understand the code or an intermediate developer wanting to build upon it, this guide will walk you through how Air Canvas uses your web camera and your hands to draw, move, and erase objects on a virtual canvas.

---

## 1. High-Level Flow 🚁

Imagine Air Canvas as a magic mirror. Instead of physical paint and a real canvas, it tracks your hand movements through your computer's webcam and translates them into colorful strokes on your screen.

Here is how the project works from start to finish:

1. **Capture Camera Input:** The program turns on your webcam and starts taking pictures (frames) continuously, usually at 30 or 60 frames per second.
2. **Find Your Hands:** Each picture is passed to an AI library called **MediaPipe**. MediaPipe acts like an X-ray vision tool; it scans the picture and finds 21 specific "landmarks" (dots) on your hand, like your fingertips, knuckles, and wrist.
3. **Recognize Gestures:** The code connects these dots to understand what your hand is doing. For example, if only the index fingertip is higher than the rest of the fingers, it knows you are pointing.
4. **Take Action:** Depending on the gesture:
   - **Pointing:** Draws colorful lines on the virtual "canvas."
   - **Open Palm:** Acts like a big eraser to clear mistakes.
   - **Pinching (3 fingers):** Selects an existing drawing.
5. **Update Canvas:** Anything you draw goes onto a blank invisible "canvas."
6. **Blend and Render UI:** Finally, the program takes the webcam feed, perfectly overlaps the invisible canvas on top of it, and draws beautiful UI elements (like the color palette and active mode) on top of everything. The blended image is then shown to you.

---

## 2. Module-Wise Explanation 📁

To make the code cleaner and easier to understand, the Air Canvas project is divided into several focused "modules" (files). Think of modules like different departments in a company working together.

### [config.py](file:///c:/Users/brahm/Desktop/air%20canvas/config.py) (The Rulebook)
This file holds all the settings (configurations) for the application.
- **What it does:** It stores camera settings, brush sizes, how strict the shape detection is, and the exact color combinations (like purple, blue, green).
- **Why we need it:** If you want to change the default brush size or add a new custom color, you only need to change it here instead of hunting through hundreds of lines of code.

### [gesture_detection.py](file:///c:/Users/brahm/Desktop/air%20canvas/gesture_detection.py) (The Eyes & Brain)
This handles MediaPipe (the AI tool) and understands what your hand is doing.
- **What it does:** It looks at the 21 dots on your hand and decides if a finger is "up" or "down". It then determines your active mode (e.g., DRAW, ERASE, MOVE).
- **Why we need it:** It translates raw math (coordinates) into human actions ("The user is drawing!").

### [shape_recognition.py](file:///c:/Users/brahm/Desktop/air%20canvas/shape_recognition.py) (The Geometry Teacher)
When you draw something, you can choose to freeze for 1.5 seconds, and this module will guess what you drew.
- **What it does:** It uses OpenCV (an image processing tool) to analyze the squiggly lines you drew. It checks angles and corner counts to convert messy strokes into perfect Circles, Rectangles, or Triangles.
- **Why we need it:** It acts as an auto-correction tool for geometry.

### [canvas_logic.py](file:///c:/Users/brahm/Desktop/air%20canvas/canvas_logic.py) (The Art Manager)
This file manages the actual data of what has been drawn.
- **What it does:** It remembers every line and shape you've created. If you want to grab a shape and move it, this file calculates the bounding box (an invisible rectangle around your drawing) so you can "pick it up".
- **Why we need it:** Without this, your drawings would be permanently stuck where you first put them.

### [ui_rendering.py](file:///c:/Users/brahm/Desktop/air%20canvas/ui_rendering.py) (The Interior Designer)
This is purely for visual aesthetics.
- **What it does:** It draws the transparent buttons, the floating color palette on the left, the brush size meter, and the status text (DRAW/MOVE/ERASE) that make the app look modern.
- **Why we need it:** It makes the application look like a premium piece of software rather than a barebones script.

### [air.py](file:///c:/Users/brahm/Desktop/air%20canvas/air.py) (The CEO / Main Entry Point)
This is the core loop that glues everything together.
- **What it does:** It starts the webcam, calls the "Eyes" to detect hands, calls the "Art Manager" to draw lines, asks the "Interior Designer" to add the final polish, and shows everything on screen.

---

## 3. Gesture Mapping ✋

Here is a breakdown of how the program translates your hand movements:

| Gesture | Finger State | Action | How Code Understands It |
| :--- | :--- | :--- | :--- |
| **Pointing** | Index UP (Others DOWN) | **Draw** | Code checks if Index Tip (Dot #8) is higher than Index Knuckle (Dot #6), while other tips are lower than their knuckles. |
| **Two Fingers** | Index & Middle UP | **Select Color** | Checks if both Index (Dot #8) and Middle (Dot #12) are high. Used to pick colors from the side toolbar. |
| **Three Fingers** | Index, Middle, Ring UP | **Select Object** | Used to highlight an already-drawn shape to get ready to move it. |
| **Open Palm** | All Fingers UP | **Erase** | Checks if all 4 fingers and your thumb are extended outward. Triggers erasing mode. |

---

## 4. Code Logic Explanation 🧩

Let's look at one of the most important functions as an example: [process_gestures()](file:///c:/Users/brahm/Desktop/air%20canvas/gesture_detection.py#59-92) located in [gesture_detection.py](file:///c:/Users/brahm/Desktop/air%20canvas/gesture_detection.py).

* **What it does:** Every fraction of a second, this function takes the coordinates of your hand from MediaPipe and decides your mode.
* **Simple Step-by-Step Logic:**
  1. *Check Palm:* Are all fingers up? If yes -> Return `"ERASE"` and the center position of the palm.
  2. *Find Index Finger:* Get the exact [(x, y)](file:///c:/Users/brahm/Desktop/air%20canvas/config.py#1-46) location of the index fingertip. Smooth out the coordinates (so jittery hands draw smooth lines).
  3. *Check Three Fingers:* Are Index, Middle, and Ring fingers up? If yes -> Return `"MOVE_SELECT"` (User is hovering over a shape).
  4. *Check Two Fingers:* Are Index and Middle up? If yes -> Return `"SELECT"` (User is pointing at the UI).
  5. *Check One Finger:* Is just the Index finger up? If yes -> Return `"DRAW"` and tell the canvas to splash color at that spot.

---

## 5. Data Structures & Storage 💾

How does the computer remember what you drew?

- **`current_stroke`: A simple List**
  As you drag your finger, the code saves every tiny coordinate into a list like this: `[(x1, y1, color), (x2, y2, color)...]`. This is your current pen stroke.
- **`strokes`: A List of Lists**
  When you lift your finger, that `current_stroke` list is saved into a master list called `strokes`. By remembering every stroke separately, we can hit "Undo" (just delete the last list in `strokes`!) or select specific objects.
- **`redo_stack`: A Backup List**
  If you click "Undo", the deleted stroke is temporarily pushed to a `redo_stack`. If you change your mind and hit "Redo", it puts it back.

---

## 6. UI & Canvas Details 🎨

The beauty of Air Canvas is that the UI feels alive.
- **The "Canvas":** Behind the scenes, the canvas is a completely black `numpy` array (just a massive spreadsheet of pixels). When you draw, the code turns these black pixels into beautifully colored pixels.
- **Transparency:** When blending the webcam feed and the black canvas, the code uses a function called `cv2.addWeighted()`. It effectively says: "Show 65% of the drawing canvas and 35% of the camera feed." Since black equals transparency, the drawings vividly pop out over your room.
- **Floating Toolbar:** The system checks your finger's [(x,y)](file:///c:/Users/brahm/Desktop/air%20canvas/config.py#1-46) position. If your finger is over the top-left area where the buttons are rendered, it actively changes your `current_color` to match whatever circle you touched.

---

## 7. Visual Flow Diagram 🔄

```ascii
[ Webcam (Starts) ]
        |
        v
[ Capture Frame (Image) ]-----\
        |                     | (Pass to UI)
        v                     v
[ MediaPipe (Finds Hand) ]  [ Draw Floating Buttons ]
        |
        v
[ Gesture Logic (What is hand doing?) ]
   /      |            \           \
(Draw)  (Erase)     (Select Color)  (Move Object)
  |       |              |             |
  v       v              v             v
[ Update Internal Canvas Memory (The 'strokes' list) ]
        |
        v
[ Blend Canvas exactly on top of the Webcam Image  ]
        |
        v
[ Show Final Result on Screen ] --> Loop repeats 30 times a second!
```

---

## 8. Tips for Beginners 🎓

If you want to poke around in this code, here are a few things to keep in mind:

- **OpenCV Flips Colors:** Most programs use RGB (Red, Green, Blue). OpenCV gets quirky and uses BGR (Blue, Green, Red). If your blue paint comes out red, this is why!
- **Coordinates Start Top-Left:** In math class, [(0,0)](file:///c:/Users/brahm/Desktop/air%20canvas/config.py#1-46) is the bottom-left. In computer graphics, [(0,0)](file:///c:/Users/brahm/Desktop/air%20canvas/config.py#1-46) is the top-left of the window. Moving down increases `y`, moving right increases `x`.
- **MediaPipe Normalizes Coordinates:** MediaPipe gives you decimals like `0.5` instead of pixels like `500`. To find the exact pixel, you have to multiply `0.5` by your screen width. (e.g., `x_pixel = mediapipe_x * screen_width`).

---

## 9. Enhancement Suggestions 🚀

Looking to take this project further? Here are some fun ways to upgrade the code:

- **Text Insertion:** Detect a unique gesture (like pointing with a pinky) that lets you type text onto the screen using a keyboard popup interface.
- **Save Video:** Currently, the code has a [save_canvas()](file:///c:/Users/brahm/Desktop/air%20canvas/canvas_logic.py#117-128) function for images. You can upgrade this using `cv2.VideoWriter()` to record a time-lapse of your drawing process.
- **Performance Boost:** As the `strokes` array gets thousands of points, it might slow down older computers. You can optimize this by "flattening" older strokes down into a single static background image instead of redrawing every single coordinate list 30 times a second.
- **Hand Tracking Upgrades:** Upgrade the [process_gestures()](file:///c:/Users/brahm/Desktop/air%20canvas/gesture_detection.py#59-92) logic to scale brush sizes based on your hand's distance from the camera (e.g., if hand is further away via bounding box size, brush gets smaller).
