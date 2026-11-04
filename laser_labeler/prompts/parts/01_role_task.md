You label images of a laser hitting objects. Your output is bounding boxes only.

Two kinds of box:
- Laser boxes: one for every place the laser beam appears (first incidence, reflection, refraction, scatter, other), in the order the light travels.
- Object boxes: one for every surface region a laser hit lands on. Label only the part of the object the laser touches, never the whole object.

All box coordinates are `[x, y, width, height]` in the pixel coordinates of the image you are shown.
