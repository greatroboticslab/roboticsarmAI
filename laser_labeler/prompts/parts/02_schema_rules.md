## Labeling rules

1. Follow the beam through every bounce, refraction and scatter, in order. Give each place the beam appears its own laser box with an event type from the allowed list. The first laser box has no `from_id`; every later one names the box before it.
   Event types: `first_incidence` is a direct hit, where the beam lands on a surface straight from the laser (two separate direct hits on the same object are both `first_incidence`). `reflection`, `refraction`, `scatter` and `other` are light that has already bounced, passed through something or been scattered and is still visible. Label every visible spot, including the glow or spread of a spot, as one of these; never leave a visible spot out.
2. For each laser hit on a surface, give one object box covering only where the laser lands on that single material and color region. If one hit spans two materials or colors, make two boxes. Example of the splitting rule only (it says nothing about what is in this image): if a laser lands on both the glass body and the ceramic handle of one cup, that is two object boxes, each with its own full attributes.
6. Name what you actually see. Never choose the nearest-sounding taxonomy entry to avoid asking: a wrong confident name is worse than a question. If the object is not in the taxonomy, ask (see the questions section) and set a low confidence.
3. Each object box has: object, material, color, incidence. These describe the box region only, not the whole object. The object must be one of the objects in the taxonomy below; for material and color, prefer the listed values but you may use a plain one-word value of your own.
4. Incidence says how the laser reaches that region: `first`, `reflected`, `refracted`, `scattered` or `other`.
5. Table rule (`{{table_rule}}`): label the table only if no other object in the image has a laser hit on it. If any other object is hit, make no table box.
6. Grayscale images have no color: write `{{color_placeholder}}` as the color.
7. Never invent measurements. All laser numbers (intensity, area, scatter, angle and so on) are computed by the app from your boxes. Do not output any.
8. Color is checked against the pixels by the app. Give your best answer, but do not agonise over it.
9. If the laser is not visible, do not guess: ask the user to mark where it first hit (see the questions section).
