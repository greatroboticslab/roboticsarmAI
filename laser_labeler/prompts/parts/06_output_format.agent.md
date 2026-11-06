## Final answer format

When you are done, reply with ONLY one JSON object (no prose, no code fence):

{
  "status": "ok" or "needs_input",
  "laser_boxes": [
    {"id": "l1", "kind": "first_incidence", "box": [x, y, w, h], "order": 0, "from_id": null, "notes": ""}
  ],
  "object_boxes": [
    {"id": "o1", "box": [x, y, w, h], "general_object": "<the object, from the taxonomy>",
     "material": "<from the taxonomy>", "finish": null, "color": "black", "incidence": "first",
     "laser_ids": ["l1"], "confidence": 0.8, "notes": ""}
  ],
  "questions": [
    {"id": "q1", "text": "...", "target_id": "o1", "field": "material", "choices": [], "allow_taxonomy_add": false}
  ]
}

`kind` is one of: first_incidence, reflection, refraction, scatter, other. `order` counts from 0 along the beam path. `confidence` is 0 to 1. Use `"status": "needs_input"` when any question is listed.
