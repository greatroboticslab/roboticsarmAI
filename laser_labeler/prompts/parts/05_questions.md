## When to ask the user

Ask a question instead of guessing when:
- the laser is not visible, or you cannot tell where it first hit;
- an object, sub-object or material is not in the taxonomy, or several fit and you cannot choose;
- the finish matters (matte vs glossy) and you cannot tell;
- lighting makes a color ambiguous;
- the beam path between two spots is not visible.

A question never stops the run. The image is parked, the user answers later, and the work resumes. So: give your best partial result and list the questions in the `questions` array.

Each question has a short `text`, optionally the `target_id` of the box it concerns, the `field` name, a list of `choices` when there are clear options, and `allow_taxonomy_add: true` when a new taxonomy entry might be the right answer.
