# Hand_mouse
This program lets you control your computer using hand gestures via your webcam.

Show an open palm (five fingers) to display the on-screen menu where you can
choose between full mouse control, an on-screen keyboard, or sleep mode. In
sleep mode the camera is processed at 1 frame per second until you show your
hand again.

The menu and keyboard overlays are drawn with OpenCV so the program works even
on systems without Tkinter installed.

Commands are executed only when the palm faces the camera; sideways gestures are
ignored.

