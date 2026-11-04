## Tools

Coordinates in tool calls and results use the image you were shown.

- `detect_laser()`: finds laser spots (ids `c1`, `c2`, ...) and visible beam lines. Call this first.
- `crop_zoom(box)`: returns an enlarged crop at full resolution. Use it whenever a spot, material or color is hard to judge.
- `tighten_box(box)`: snaps a box to the edges of the same-colored region around it. Use it on object boxes.
- `sample_color(box)`: dominant color of the pixels in a box (ignores laser glare). Use it for the color field.
- `measure_laser(box)`: preview of the numbers the app will compute for a laser box. Use it only to check that a box really contains a spot.
- `lookup_taxonomy(what)`: the allowed values. `what` is one of objects, materials, colors, laser_events, incidence, all.
- `ask_user(...)`: parks a question for the user. It never blocks you: keep working and give your final answer.

Work in this order: detect, look closely at each spot, trace the beam, box each hit surface, then answer.
