## How this task works

The app has already found the laser spots with image processing. You are shown the whole image with each candidate spot boxed and labeled (`c1`, `c2`, ...), followed by one zoomed crop per candidate. You do not draw boxes. For each candidate, decide what it is and what surface it lands on.

## Answer format

Reply with ONLY one JSON object (no prose, no code fence):

{
  "status": "ok" or "needs_input",
  "candidates": [
    {"id": "c1", "is_laser": true, "kind": "first_incidence", "order": 0, "from_id": null,
     "object": {"general_object": "<the object, from the taxonomy>", "material": "<from the taxonomy>",
                "finish": null, "incidence": "first", "confidence": 0.8},
     "notes": ""}
  ],
  "questions": [
    {"id": "q1", "text": "...", "target_id": "c1", "field": "material", "choices": [], "allow_taxonomy_add": false}
  ]
}

Include every candidate id you were shown. Set `is_laser` to false for glare, reflections of room lights or anything that is not the laser (then `kind`, `order` and `object` can be null). `kind` is one of: first_incidence, reflection, refraction, scatter, other. `from_id` is the candidate id the beam came from (null for the first). `object` describes only the surface region the spot lands on; use null if it lands on nothing you can label. Do not give a color: the app measures it.
